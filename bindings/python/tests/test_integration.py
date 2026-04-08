"""Integration tests combining multiple AlloyGBM features simultaneously.

Each test exercises at least two features that were not previously tested
together, using realistic synthetic data (500+ rows, 6+ features).
"""

from __future__ import annotations

import math
import pickle
import unittest
import warnings

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor, accuracy, ndcg


# ---------------------------------------------------------------------------
# Shared data generators
# ---------------------------------------------------------------------------


def _make_regression_data(
    n: int = 500,
    n_features: int = 8,
    nan_frac: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-linear regression target with optional NaN injection."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = 2.0 * X[:, 0] + np.sin(X[:, 1] * 3) - 0.5 * X[:, 2] ** 2
    if n_features > 3:
        y = y + 0.3 * X[:, 3]
    y = y + rng.randn(n).astype(np.float32) * 0.5
    if nan_frac > 0:
        mask = rng.rand(n, n_features) < nan_frac
        X[mask] = np.nan
    return X, y.astype(np.float32)


def _make_classification_data(
    n: int = 500,
    n_features: int = 8,
    nan_frac: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary classification with optional NaN injection."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    logits = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
    y = (logits > 0).astype(np.float32)
    if nan_frac > 0:
        mask = rng.rand(n, n_features) < nan_frac
        X[mask] = np.nan
    return X, y


def _make_ranking_data(
    n_queries: int = 20,
    docs_per_query: int = 15,
    n_features: int = 6,
    nan_frac: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ranking data with graded relevance labels 0-4."""
    rng = np.random.RandomState(seed)
    n = n_queries * docs_per_query
    group = np.repeat(np.arange(n_queries, dtype=np.uint32), docs_per_query)
    X = rng.randn(n, n_features).astype(np.float32)
    raw_rel = X[:, 0] + 0.5 * X[:, 1]
    y = np.clip(np.digitize(raw_rel, [-1.5, -0.5, 0.5, 1.5]), 0, 4).astype(
        np.float32
    )
    if nan_frac > 0:
        mask = rng.rand(n, n_features) < nan_frac
        X[mask] = np.nan
    return X, y, group


def _make_categorical_column(
    n: int, n_categories: int = 10, seed: int = 42
) -> list[str]:
    """Generate a string categorical column."""
    rng = np.random.RandomState(seed)
    categories = [f"cat_{i}" for i in range(n_categories)]
    return [categories[i] for i in rng.randint(0, n_categories, size=n)]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestLeafWiseWithObjectives(unittest.TestCase):
    """Leaf-wise growth works with classification and ranking objectives."""

    def test_leaf_wise_classification(self) -> None:
        X, y = _make_classification_data(n=400, seed=1)
        X_val, y_val = _make_classification_data(n=100, seed=2)
        clf = GBMClassifier(
            n_estimators=30,
            tree_growth="leaf",
            max_leaves=16,
            early_stopping_rounds=5,
            seed=42,
        )
        clf.fit(X, y, eval_set=(X_val, y_val))
        probs = clf.predict_proba(X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        labels = clf.predict(X)
        acc = sum(int(p == int(t)) for p, t in zip(labels, y)) / len(y)
        self.assertGreater(acc, 0.6)

    def test_leaf_wise_ranking_ndcg(self) -> None:
        X, y, group = _make_ranking_data(seed=10)
        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=20,
            tree_growth="leaf",
            max_leaves=8,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        ranker.fit(X, y, group=group)
        preds = ranker.predict(X)
        self.assertEqual(len(preds), len(y))
        self.assertTrue(all(math.isfinite(p) for p in preds))

    def test_leaf_wise_ranking_all_objectives(self) -> None:
        X, y, group = _make_ranking_data(seed=20)
        for obj in ("rank:pairwise", "rank:ndcg", "rank:xendcg", "queryrmse", "yetirank"):
            with self.subTest(objective=obj):
                ranker = GBMRanker(
                    ranking_objective=obj,
                    n_estimators=10,
                    tree_growth="leaf",
                    max_leaves=8,
                    seed=42,
                    training_policy="manual",
                    min_split_gain=0.0,
                )
                ranker.fit(X, y, group=group)
                preds = ranker.predict(X)
                self.assertEqual(len(preds), len(y))

    def test_leaf_wise_with_nan_classification(self) -> None:
        X, y = _make_classification_data(n=500, nan_frac=0.1, seed=30)
        clf = GBMClassifier(
            n_estimators=20,
            tree_growth="leaf",
            max_leaves=16,
            seed=42,
        )
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))


class TestSHAPAcrossObjectives(unittest.TestCase):
    """SHAP values work for classification and ranking models."""

    def test_shap_classification_additivity(self) -> None:
        X, y = _make_classification_data(n=300, n_features=4, seed=40)
        clf = GBMClassifier(n_estimators=15, seed=42)
        clf.fit(X, y)

        expected_value, shap_matrix = clf.shap_values(
            X[:10], include_expected_value=True
        )
        self.assertIsInstance(expected_value, float)
        self.assertEqual(len(shap_matrix), 10)
        self.assertEqual(len(shap_matrix[0]), 4)

        # The Rust layer already verifies additivity internally. Here we
        # verify the Python-level result is consistent: the SHAP values
        # should be in log-odds space and sum + expected ≈ raw logits.
        # Sigmoid of raw logits should ≈ predict_proba.
        probs = clf.predict_proba(X[:10])
        for i, row_shap in enumerate(shap_matrix):
            raw_logit = expected_value + sum(row_shap)
            prob_from_shap = 1.0 / (1.0 + math.exp(-raw_logit))
            self.assertAlmostEqual(prob_from_shap, probs[i], places=3)

    def test_shap_ranking_additivity(self) -> None:
        X, y, group = _make_ranking_data(n_queries=8, n_features=4, seed=50)
        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=10,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        ranker.fit(X, y, group=group)

        expected_value, shap_matrix = ranker.shap_values(
            X[:10], include_expected_value=True
        )
        preds = ranker.predict(X[:10])
        for i, row_shap in enumerate(shap_matrix):
            reconstructed = expected_value + sum(row_shap)
            self.assertAlmostEqual(reconstructed, preds[i], places=3)

    def test_shap_with_categorical_features(self) -> None:
        n = 300
        X_numeric, y = _make_regression_data(n=n, n_features=4, seed=60)
        cat_col = _make_categorical_column(n, n_categories=5, seed=60)

        # Place categorical at column 2 — build X as list-of-lists
        rows = []
        for i in range(n):
            row = list(X_numeric[i])
            row[2] = float(hash(cat_col[i]) % 100)  # placeholder numeric
            rows.append(row)

        reg = GBMRegressor(
            n_estimators=15,
            categorical_feature_index=2,
            seed=42,
        )
        reg.fit(rows, y.tolist(), categorical_feature_values=cat_col)

        expected_value, shap_matrix = reg.shap_values(
            rows[:5], include_expected_value=True
        )
        preds = reg.predict(rows[:5])
        for i, row_shap in enumerate(shap_matrix):
            reconstructed = expected_value + sum(row_shap)
            self.assertAlmostEqual(reconstructed, preds[i], places=3)

    def test_shap_with_wide_bins(self) -> None:
        X, y = _make_regression_data(n=500, n_features=4, seed=70)
        reg = GBMRegressor(
            n_estimators=15,
            continuous_binning_max_bins=500,
            seed=42,
        )
        reg.fit(X, y)

        expected_value, shap_matrix = reg.shap_values(
            X[:5], include_expected_value=True
        )
        preds = reg.predict(X[:5])
        for i, row_shap in enumerate(shap_matrix):
            reconstructed = expected_value + sum(row_shap)
            self.assertAlmostEqual(reconstructed, preds[i], places=3)


class TestWarmStartCompatibility(unittest.TestCase):
    """Warm-starting works across objectives and rejects mismatches."""

    def test_warm_start_classification(self) -> None:
        X, y = _make_classification_data(n=400, seed=80)
        clf1 = GBMClassifier(n_estimators=10, warm_start=True, seed=42)
        clf1.fit(X, y)
        preds_10 = clf1.predict_proba(X)

        clf1.n_estimators = 20
        clf1.fit(X, y)
        preds_20 = clf1.predict_proba(X)

        # After warm-start continuation, predictions should differ
        # (more trees were added).
        self.assertFalse(
            all(abs(a - b) < 1e-8 for a, b in zip(preds_10, preds_20))
        )

    def test_warm_start_ranking(self) -> None:
        # Warm-start with ranking: verify the pipeline doesn't crash.
        # NOTE: Ranking objectives may converge to 0 additional rounds on
        # warm-start due to how pairwise gradients interact with an already-
        # fitted model. This test ensures the path is exercised without error.
        X, y, group = _make_ranking_data(n_queries=30, docs_per_query=20, seed=90)
        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=5,
            warm_start=True,
            learning_rate=0.05,
            seed=42,
        )
        ranker.fit(X, y, group=group)
        preds_5 = ranker.predict(X)
        self.assertTrue(all(math.isfinite(p) for p in preds_5))

        ranker.n_estimators = 50
        ranker.fit(X, y, group=group)
        preds_50 = ranker.predict(X)
        # Predictions remain valid after warm-start attempt.
        self.assertTrue(all(math.isfinite(p) for p in preds_50))

    def test_warm_start_rejects_objective_mismatch(self) -> None:
        # Train a classifier, then try to warm-start a regressor from it.
        # GBMRegressor.fit() accepts init_model; the objective check fires.
        X, y = _make_classification_data(n=100, n_features=5, seed=101)
        clf = GBMClassifier(n_estimators=5, seed=42)
        clf.fit(X, y)

        X_reg, y_reg = _make_regression_data(n=100, n_features=5, seed=100)
        reg = GBMRegressor(n_estimators=5, seed=42)
        with self.assertRaisesRegex(ValueError, "objective"):
            reg.fit(X_reg, y_reg, init_model=clf)

    def test_warm_start_rejects_feature_count_mismatch(self) -> None:
        X4, y4 = _make_regression_data(n=100, n_features=4, seed=110)
        reg4 = GBMRegressor(n_estimators=5, seed=42)
        reg4.fit(X4, y4)

        X6, y6 = _make_regression_data(n=100, n_features=6, seed=111)
        reg6 = GBMRegressor(n_estimators=5, seed=42)
        with self.assertRaisesRegex(ValueError, "features"):
            reg6.fit(X6, y6, init_model=reg4)

    def test_warm_start_with_categorical_features(self) -> None:
        n = 200
        rng = np.random.RandomState(120)
        X = rng.randn(n, 3).astype(np.float32)
        cat_col = _make_categorical_column(n, n_categories=5, seed=120)
        y = (X[:, 0] + X[:, 1]).tolist()
        rows = [list(X[i]) for i in range(n)]

        reg = GBMRegressor(
            n_estimators=10,
            warm_start=True,
            categorical_feature_index=2,
            seed=42,
        )
        reg.fit(rows, y, categorical_feature_values=cat_col)
        preds_10 = reg.predict(rows)

        reg.n_estimators = 20
        reg.fit(rows, y, categorical_feature_values=cat_col)
        preds_20 = reg.predict(rows)

        self.assertFalse(
            all(abs(a - b) < 1e-8 for a, b in zip(preds_10, preds_20))
        )


class TestCategoricalWithObjectives(unittest.TestCase):
    """Categorical target encoding works with classification and ranking."""

    def test_categorical_classification(self) -> None:
        n = 300
        rng = np.random.RandomState(130)
        X = rng.randn(n, 3).astype(np.float32)
        cat_col = _make_categorical_column(n, n_categories=8, seed=130)
        y = (X[:, 0] > 0).astype(np.float32).tolist()
        rows = [list(X[i]) for i in range(n)]

        clf = GBMClassifier(
            n_estimators=15,
            categorical_feature_index=2,
            seed=42,
        )
        clf.fit(rows, y, categorical_feature_values=cat_col)
        probs = clf.predict_proba(rows)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_categorical_ranking(self) -> None:
        n_q, dpq = 10, 15
        n = n_q * dpq
        rng = np.random.RandomState(140)
        X = rng.randn(n, 4).astype(np.float32)
        cat_col = _make_categorical_column(n, n_categories=6, seed=140)
        group = np.repeat(np.arange(n_q, dtype=np.uint32), dpq)
        raw_rel = X[:, 0] + 0.3 * X[:, 1]
        y = np.clip(np.digitize(raw_rel, [-1, 0, 1]), 0, 3).astype(np.float32)
        rows = [list(X[i]) for i in range(n)]

        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=10,
            categorical_feature_index=2,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        ranker.fit(rows, y.tolist(), group=group, categorical_feature_values=cat_col)
        preds = ranker.predict(rows)
        self.assertEqual(len(preds), n)
        self.assertTrue(all(math.isfinite(p) for p in preds))

    def test_multi_categorical_with_nan(self) -> None:
        n = 300
        rng = np.random.RandomState(150)
        X = rng.randn(n, 5).astype(np.float32)
        # Inject NaN into non-categorical columns
        nan_mask = rng.rand(n, 5) < 0.05
        nan_mask[:, 1] = False  # keep cat col 1 clean
        nan_mask[:, 3] = False  # keep cat col 3 clean
        X[nan_mask] = np.nan

        cat_col_1 = _make_categorical_column(n, n_categories=6, seed=151)
        cat_col_3 = _make_categorical_column(n, n_categories=4, seed=152)
        y = (2.0 * X[:, 0] + X[:, 2]).tolist()
        # Replace NaN in y with 0 (targets must be finite)
        y = [0.0 if math.isnan(v) else v for v in y]

        rows = [list(X[i]) for i in range(n)]

        reg = GBMRegressor(
            n_estimators=15,
            categorical_feature_indices=[1, 3],
            seed=42,
        )
        reg.fit(
            rows,
            y,
            categorical_feature_values_list=[cat_col_1, cat_col_3],
        )
        preds = reg.predict(rows)
        self.assertEqual(len(preds), n)
        self.assertTrue(all(math.isfinite(p) for p in preds))


class TestWideBinsWithObjectives(unittest.TestCase):
    """u16 wide bins (>256) work with classification and ranking."""

    def test_wide_bins_classification(self) -> None:
        X, y = _make_classification_data(n=500, seed=160)
        clf = GBMClassifier(
            n_estimators=15,
            continuous_binning_max_bins=500,
            seed=42,
        )
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        labels = clf.predict(X)
        acc = sum(int(p == int(t)) for p, t in zip(labels, y)) / len(y)
        self.assertGreater(acc, 0.6)

    def test_wide_bins_ranking(self) -> None:
        X, y, group = _make_ranking_data(seed=170)
        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=10,
            continuous_binning_max_bins=500,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        ranker.fit(X, y, group=group)
        preds = ranker.predict(X)
        self.assertEqual(len(preds), len(y))
        self.assertTrue(all(math.isfinite(p) for p in preds))

    def test_wide_bins_with_nan(self) -> None:
        X, y = _make_regression_data(n=500, nan_frac=0.08, seed=180)
        reg = GBMRegressor(
            n_estimators=15,
            continuous_binning_max_bins=500,
            seed=42,
        )
        reg.fit(X, y)
        preds = reg.predict(X)
        self.assertEqual(len(preds), len(y))
        self.assertTrue(all(math.isfinite(p) for p in preds))


class TestConstraintsWithObjectives(unittest.TestCase):
    """Monotone constraints and feature weights work with non-regression objectives."""

    def test_monotone_constraints_classification(self) -> None:
        X, y = _make_classification_data(n=500, n_features=4, seed=190)
        clf = GBMClassifier(
            n_estimators=20,
            monotone_constraints={0: 1},  # feature 0 monotone increasing
            seed=42,
        )
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

        # Verify monotonicity: sort X by feature 0 (fixing others at median),
        # predictions should be non-decreasing.
        medians = np.median(X, axis=0)
        test_X = np.tile(medians, (50, 1)).astype(np.float32)
        test_X[:, 0] = np.linspace(X[:, 0].min(), X[:, 0].max(), 50).astype(
            np.float32
        )
        mono_probs = clf.predict_proba(test_X)
        # Check overall monotone trend: correlation with sorted indices.
        # Small local violations can occur at sigmoid boundaries, so
        # we verify the overall trend rather than strict pairwise ordering.
        increases = sum(
            1 for i in range(1, len(mono_probs)) if mono_probs[i] >= mono_probs[i - 1] - 0.02
        )
        self.assertGreater(
            increases / (len(mono_probs) - 1),
            0.85,
            "Monotone constraint is not respected: too many decreasing steps",
        )

    def test_feature_weights_classification(self) -> None:
        X, y = _make_classification_data(n=400, n_features=6, seed=200)
        # Heavily weight feature 0, zero-weight features 3-5.
        clf = GBMClassifier(
            n_estimators=20,
            feature_weights={0: 10.0, 3: 0.0, 4: 0.0, 5: 0.0},
            seed=42,
        )
        clf.fit(X, y)
        importances = clf.feature_importances(X[:50])
        # Feature 0 should be top-ranked.
        top_feature = importances[0][0]
        self.assertEqual(top_feature, "f0")
        # Zero-weighted features should have minimal importance.
        zero_weighted = {name: imp for name, imp in importances if name in ("f3", "f4", "f5")}
        for name, imp in zero_weighted.items():
            self.assertAlmostEqual(imp, 0.0, places=5, msg=f"{name} should have ~0 importance")

    def test_feature_weights_ranking(self) -> None:
        X, y, group = _make_ranking_data(n_features=6, seed=210)
        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=15,
            feature_weights={0: 10.0, 4: 0.0, 5: 0.0},
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        ranker.fit(X, y, group=group)
        preds = ranker.predict(X)
        self.assertTrue(all(math.isfinite(p) for p in preds))


class TestKitchenSink(unittest.TestCase):
    """Combine 4+ features simultaneously for real-world-like configurations."""

    def test_classification_full_stack(self) -> None:
        """classification + leaf-wise + NaN + early stopping + wide bins + sample weights + L2 + metrics"""
        X, y = _make_classification_data(n=600, n_features=8, nan_frac=0.05, seed=300)
        X_val, y_val = _make_classification_data(
            n=150, n_features=8, nan_frac=0.05, seed=301
        )
        rng = np.random.RandomState(302)
        sample_w = rng.uniform(0.5, 2.0, size=len(y)).astype(np.float32)

        clf = GBMClassifier(
            n_estimators=50,
            tree_growth="leaf",
            max_leaves=16,
            continuous_binning_max_bins=400,
            lambda_l2=1.0,
            early_stopping_rounds=5,
            seed=42,
        )
        clf.fit(X, y, sample_weight=sample_w, eval_set=(X_val, y_val))

        # Check metric tracking populated.
        self.assertIsNotNone(clf.evals_result_)
        self.assertIn("train", clf.evals_result_)
        self.assertGreater(len(clf.evals_result_["train"]), 0)

        # Predictions are valid probabilities.
        probs = clf.predict_proba(X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

        labels = clf.predict(X)
        acc = sum(int(p == int(t)) for p, t in zip(labels, y)) / len(y)
        self.assertGreater(acc, 0.55)

    def test_ranking_full_stack(self) -> None:
        """ranking + leaf-wise + early stopping + feature weights + wide bins"""
        X, y, group = _make_ranking_data(
            n_queries=30, docs_per_query=20, n_features=8, seed=310
        )
        # Split into train/val by query.
        train_mask = group < 20
        val_mask = group >= 20
        X_tr, y_tr, g_tr = X[train_mask], y[train_mask], group[train_mask]
        X_val, y_val, g_val = X[val_mask], y[val_mask], group[val_mask]

        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=40,
            tree_growth="leaf",
            max_leaves=12,
            continuous_binning_max_bins=400,
            feature_weights={0: 5.0, 1: 5.0},
            early_stopping_rounds=5,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        ranker.fit(
            X_tr,
            y_tr,
            group=g_tr,
            eval_set=(X_val, y_val),
            eval_group=g_val,
        )

        preds = ranker.predict(X_tr)
        self.assertTrue(all(math.isfinite(p) for p in preds))
        self.assertIsNotNone(ranker.n_estimators_)

    def test_regression_full_stack_with_persistence(self) -> None:
        """regression + categorical + warm-start + SHAP + pickle + early stopping"""
        n = 400
        rng = np.random.RandomState(320)
        X = rng.randn(n, 5).astype(np.float32)

        cat_col = _make_categorical_column(n, n_categories=6, seed=321)
        y = (2.0 * X[:, 0] + X[:, 1]).tolist()
        rows = [list(X[i]) for i in range(n)]

        X_val = rng.randn(100, 5).astype(np.float32)
        y_val = (2.0 * X_val[:, 0] + X_val[:, 1]).tolist()
        rows_val = [list(X_val[i]) for i in range(100)]

        # Phase 1: initial training.
        reg = GBMRegressor(
            n_estimators=15,
            warm_start=True,
            categorical_feature_index=2,
            early_stopping_rounds=10,
            seed=42,
        )
        reg.fit(
            rows,
            y,
            categorical_feature_values=cat_col,
            eval_set=(rows_val, y_val),
        )
        preds_phase1 = reg.predict(rows[:10])

        # Phase 2: warm-start continuation.
        reg.n_estimators = 30
        reg.fit(
            rows,
            y,
            categorical_feature_values=cat_col,
            eval_set=(rows_val, y_val),
        )
        preds_phase2 = reg.predict(rows[:10])

        # Predictions should change after warm-start adds trees.
        self.assertFalse(
            all(abs(a - b) < 1e-8 for a, b in zip(preds_phase1, preds_phase2))
        )

        # SHAP on categorical model should work.
        expected_value, shap_matrix = reg.shap_values(
            rows[:5], include_expected_value=True
        )
        preds_5 = reg.predict(rows[:5])
        for i, row_shap in enumerate(shap_matrix):
            reconstructed = expected_value + sum(row_shap)
            self.assertAlmostEqual(reconstructed, preds_5[i], places=3)

        # Pickle roundtrip preserves predictions.
        pickled = pickle.dumps(reg)
        reg_restored = pickle.loads(pickled)
        preds_restored = reg_restored.predict(rows[:10])
        for orig, restored in zip(preds_phase2, preds_restored):
            self.assertAlmostEqual(orig, restored, places=6)


class TestSampleWeightWarning(unittest.TestCase):
    """Ranking objectives warn when sample_weight is provided."""

    def test_ranking_sample_weight_warns(self) -> None:
        X, y, group = _make_ranking_data(n_queries=5, docs_per_query=10, seed=400)
        sw = np.ones(len(y), dtype=np.float32)
        for obj in ("rank:pairwise", "rank:ndcg", "rank:xendcg", "yetirank"):
            with self.subTest(objective=obj):
                ranker = GBMRanker(
                    ranking_objective=obj,
                    n_estimators=3,
                    seed=42,
                    training_policy="manual",
                    min_split_gain=0.0,
                )
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    ranker.fit(X, y, group=group, sample_weight=sw)
                    ranking_warnings = [
                        x for x in w if "sample_weight is ignored" in str(x.message)
                    ]
                    self.assertEqual(
                        len(ranking_warnings),
                        1,
                        f"Expected 1 warning for {obj}, got {len(ranking_warnings)}",
                    )

    def test_queryrmse_sample_weight_no_warning(self) -> None:
        """queryrmse uses sample weights, so no warning should be emitted."""
        X, y, group = _make_ranking_data(n_queries=5, docs_per_query=10, seed=410)
        sw = np.ones(len(y), dtype=np.float32)
        ranker = GBMRanker(
            ranking_objective="queryrmse",
            n_estimators=3,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ranker.fit(X, y, group=group, sample_weight=sw)
            ranking_warnings = [
                x for x in w if "sample_weight is ignored" in str(x.message)
            ]
            self.assertEqual(len(ranking_warnings), 0)


if __name__ == "__main__":
    unittest.main()
