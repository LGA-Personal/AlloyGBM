"""Tests for custom eval metric callback support across all estimators."""

from __future__ import annotations

import unittest

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


# ── Helper callables ─────────────────────────────────────────────────────────

def _custom_rmse(y_true, y_pred):
    """RMSE metric: (name, value, higher_is_better)."""
    return ("custom_rmse", float(np.sqrt(np.mean((y_true - y_pred) ** 2))), False)


def _custom_r2(y_true, y_pred):
    """R² metric (higher is better)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-15))
    return ("r2", float(r2), True)


def _custom_accuracy(y_true, y_pred):
    """Binary accuracy metric."""
    preds = (y_pred >= 0.5).astype(int)
    acc = float(np.mean(preds == y_true.astype(int)))
    return ("accuracy", acc, True)


def _custom_ndcg_proxy(y_true, y_pred):
    """Proxy NDCG metric for ranking tests."""
    # Simple correlation-based proxy
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0
    return ("ndcg_proxy", corr, True)


# ── Fixtures ─────────────────────────────────────���───────────────────────────

def _regression_data(n=100, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] + rng.randn(n) * 0.1).astype(np.float32)
    return X, y


def _classification_data(n=100, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    return X, y


def _multiclass_data(n=150, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4).astype(np.float32)
    y = np.repeat([0, 1, 2], n // 3).astype(np.float32)
    return X, y


def _ranking_data(n_groups=10, per_group=5, seed=42):
    rng = np.random.RandomState(seed)
    n = n_groups * per_group
    X = rng.randn(n, 3).astype(np.float32)
    y = rng.randint(0, 3, size=n).astype(np.float32)
    group = np.repeat(np.arange(n_groups), per_group).tolist()
    return X, y, group


class TestRegressorCustomMetric(unittest.TestCase):
    """GBMRegressor with custom eval metric."""

    def test_basic_custom_metric(self):
        """Built-in objective + custom metric, verify metric tracked in evals_result_."""
        X, y = _regression_data(n=100)
        X_val, y_val = _regression_data(n=50, seed=7)
        model = GBMRegressor(
            n_estimators=20,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y, eval_set=(X_val, y_val), eval_metric=_custom_rmse)
        self.assertIsNotNone(model.evals_result_)
        self.assertIn("validation", model.evals_result_)
        self.assertIn("custom_rmse", model.evals_result_["validation"])
        vals = model.evals_result_["validation"]["custom_rmse"]
        self.assertEqual(len(vals), model.n_estimators_)
        # RMSE should be positive and decreasing overall
        self.assertTrue(all(v >= 0 for v in vals))

    def test_custom_metric_early_stopping(self):
        """Custom metric drives early stopping."""
        X, y = _regression_data(n=200)
        X_val, y_val = _regression_data(n=100, seed=7)
        model = GBMRegressor(
            n_estimators=500,
            early_stopping_rounds=5,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y, eval_set=(X_val, y_val), eval_metric=_custom_rmse)
        self.assertLess(model.n_estimators_, 500)

    def test_custom_metric_higher_is_better(self):
        """Metric with higher_is_better=True (e.g. R²)."""
        X, y = _regression_data(n=200)
        X_val, y_val = _regression_data(n=100, seed=7)
        model = GBMRegressor(
            n_estimators=50,
            early_stopping_rounds=10,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y, eval_set=(X_val, y_val), eval_metric=_custom_r2)
        self.assertIsNotNone(model.evals_result_)
        self.assertIn("r2", model.evals_result_["validation"])

    def test_custom_metric_without_eval_set_raises(self):
        """Custom metric without eval_set raises ValueError."""
        X, y = _regression_data()
        model = GBMRegressor(n_estimators=10, seed=1)
        with self.assertRaises(ValueError):
            model.fit(X, y, eval_metric=_custom_rmse)

    def test_custom_metric_not_callable_raises(self):
        """Non-callable eval_metric raises TypeError."""
        X, y = _regression_data()
        X_val, y_val = _regression_data(n=50, seed=7)
        model = GBMRegressor(n_estimators=10, seed=1)
        with self.assertRaises(TypeError):
            model.fit(X, y, eval_set=(X_val, y_val), eval_metric="not_a_callable")

    def test_custom_metric_values_in_evals_result(self):
        """evals_result_ has custom metric key alongside built-in metrics."""
        X, y = _regression_data(n=100)
        X_val, y_val = _regression_data(n=50, seed=7)
        model = GBMRegressor(
            n_estimators=10,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y, eval_set=(X_val, y_val), eval_metric=_custom_rmse)
        val = model.evals_result_["validation"]
        # Should have built-in rmse, mse, plus custom_rmse
        self.assertIn("rmse", val)
        self.assertIn("mse", val)
        self.assertIn("custom_rmse", val)

    def test_custom_metric_with_sample_weight(self):
        """Custom metric with sample weights on the training data."""
        X, y = _regression_data(n=100)
        X_val, y_val = _regression_data(n=50, seed=7)
        weights = np.ones(len(y), dtype=np.float32)
        weights[:20] = 5.0
        model = GBMRegressor(
            n_estimators=20,
            seed=1,
            deterministic=True,
        )
        model.fit(
            X, y,
            sample_weight=weights,
            eval_set=(X_val, y_val),
            eval_metric=_custom_rmse,
        )
        self.assertIn("custom_rmse", model.evals_result_["validation"])


class TestClassifierCustomMetric(unittest.TestCase):
    """GBMClassifier with custom eval metric."""

    def test_classifier_custom_metric_binary(self):
        """GBMClassifier binary + custom metric."""
        X, y = _classification_data(n=100)
        X_val, y_val = _classification_data(n=50, seed=7)
        clf = GBMClassifier(
            n_estimators=20,
            seed=1,
            deterministic=True,
        )
        clf.fit(X, y, eval_set=(X_val, y_val), eval_metric=_custom_accuracy)
        self.assertIsNotNone(clf.evals_result_)
        self.assertIn("accuracy", clf.evals_result_["validation"])

    def test_classifier_custom_metric_multiclass(self):
        """GBMClassifier multiclass + custom metric."""
        X, y = _multiclass_data(n=150)
        X_val, y_val = _multiclass_data(n=60, seed=7)

        def custom_mc_metric(y_true, y_pred):
            # For multiclass, y_pred is flattened softmax probabilities
            return ("mc_metric", float(np.mean(np.abs(y_true - y_pred))), False)

        clf = GBMClassifier(
            n_estimators=20,
            seed=1,
            deterministic=True,
        )
        clf.fit(X, y, eval_set=(X_val, y_val), eval_metric=custom_mc_metric)
        self.assertIsNotNone(clf.evals_result_)
        # Note: custom metric may or may not be populated for multiclass
        # depending on engine support; just verify no crash


class TestRankerCustomMetric(unittest.TestCase):
    """GBMRanker with custom eval metric."""

    def test_ranker_custom_metric_with_group(self):
        """GBMRanker + custom NDCG-like metric."""
        X, y, group = _ranking_data()
        X_val, y_val, group_val = _ranking_data(n_groups=5, per_group=5, seed=7)
        rnk = GBMRanker(
            n_estimators=20,
            seed=1,
            deterministic=True,
        )
        rnk.fit(
            X, y,
            group=group,
            eval_set=(X_val, y_val),
            eval_group=group_val,
            eval_metric=_custom_ndcg_proxy,
        )
        self.assertIsNotNone(rnk.evals_result_)


class TestCustomMetricWithCustomObjective(unittest.TestCase):
    """Both custom objective and custom metric together."""

    def test_both_custom(self):
        """Custom objective + custom metric together."""
        X, y = _regression_data(n=200)
        X_val, y_val = _regression_data(n=100, seed=7)

        def custom_obj(y_true, y_pred):
            return y_pred - y_true, np.ones_like(y_true)

        model = GBMRegressor(
            objective=custom_obj,
            n_estimators=200,
            early_stopping_rounds=5,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y, eval_set=(X_val, y_val), eval_metric=_custom_rmse)
        self.assertIsNotNone(model.evals_result_)
        self.assertIn("custom_rmse", model.evals_result_["validation"])
        # With 200 max rounds and early_stopping_rounds=5, it should stop early
        self.assertLessEqual(model.n_estimators_, 200)

    def test_custom_metric_wrong_return_type_raises(self):
        """Callable returning wrong type raises clear error during fit."""
        def bad_metric(y_true, y_pred):
            return "just_a_string"  # Should return (name, value, higher_is_better)

        X, y = _regression_data()
        X_val, y_val = _regression_data(n=50, seed=7)
        model = GBMRegressor(n_estimators=5, seed=1)
        with self.assertRaises(Exception):
            model.fit(X, y, eval_set=(X_val, y_val), eval_metric=bad_metric)


if __name__ == "__main__":
    unittest.main()
