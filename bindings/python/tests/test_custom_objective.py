"""Tests for custom objective callable support across all estimators."""

from __future__ import annotations

import pickle
import tempfile
import unittest
import warnings

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


# ── Helper callables ──────��──────────────────────────────────────────────────

def _custom_mse_objective(y_true, y_pred):
    """MSE-equivalent custom objective: (grad, hess)."""
    grad = y_pred - y_true
    hess = np.ones_like(y_true)
    return grad, hess


def _custom_logistic_objective(y_true, y_pred):
    """Logistic loss gradients for binary classification."""
    sigmoid = 1.0 / (1.0 + np.exp(-y_pred))
    grad = sigmoid - y_true
    hess = sigmoid * (1.0 - sigmoid)
    return grad, hess


def _custom_pairwise_objective(y_true, y_pred):
    """Simple pointwise proxy for ranking."""
    grad = y_pred - y_true
    hess = np.ones_like(y_true)
    return grad, hess


# ── Fixtures ─��───────────────────���───────────────────────────────────────────

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


def _ranking_data(n_groups=10, per_group=5, seed=42):
    rng = np.random.RandomState(seed)
    n = n_groups * per_group
    X = rng.randn(n, 3).astype(np.float32)
    y = rng.randint(0, 3, size=n).astype(np.float32)
    group = np.repeat(np.arange(n_groups), per_group).tolist()
    return X, y, group


class TestRegressorCustomObjective(unittest.TestCase):
    """GBMRegressor with custom objective."""

    def test_basic_custom_objective(self):
        """Custom MSE objective produces predictions."""
        X, y = _regression_data()
        model = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=20,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))
        # Predictions should be non-trivial (not all zeros)
        self.assertTrue(any(abs(p) > 0.01 for p in preds))

    def test_custom_objective_approximates_builtin(self):
        """Custom MSE objective should produce similar results to built-in."""
        X, y = _regression_data()
        # Built-in
        m_builtin = GBMRegressor(
            n_estimators=20,
            seed=1,
            deterministic=True,
            training_policy="manual",
        )
        m_builtin.fit(X, y)
        p_builtin = np.array(m_builtin.predict(X))

        # Custom (MSE-equivalent)
        m_custom = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=20,
            seed=1,
            deterministic=True,
            training_policy="manual",
        )
        m_custom.fit(X, y)
        p_custom = np.array(m_custom.predict(X))

        # They should be similar (not identical due to possible differences in
        # initial prediction or leaf refinement)
        np.testing.assert_allclose(p_custom, p_builtin, atol=0.5)

    def test_custom_objective_with_eval_set(self):
        """Custom objective with validation set populates evals_result_."""
        X, y = _regression_data(n=100)
        X_val, y_val = _regression_data(n=50, seed=7)
        model = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=20,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y, eval_set=(X_val, y_val))
        self.assertIsNotNone(model.evals_result_)
        self.assertIn("validation", model.evals_result_)

    def test_custom_objective_with_early_stopping(self):
        """Custom objective with early stopping works."""
        X, y = _regression_data(n=200)
        X_val, y_val = _regression_data(n=100, seed=7)
        model = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=200,
            early_stopping_rounds=5,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y, eval_set=(X_val, y_val))
        # Should have stopped before all 200 rounds
        self.assertLess(model.n_estimators_, 200)

    def test_custom_objective_with_sample_weight(self):
        """Custom objective honours sample weights."""
        X, y = _regression_data()
        weights = np.ones(len(y), dtype=np.float32)
        weights[:20] = 5.0  # Weight first 20 samples more heavily
        model = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=20,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y, sample_weight=weights)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))

    def test_custom_objective_warm_start(self):
        """Custom objective with warm_start=True."""
        X, y = _regression_data()
        model = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=10,
            seed=1,
            deterministic=True,
            warm_start=True,
        )
        model.fit(X, y)
        preds_first = np.array(model.predict(X))

        # Second fit should continue from previous trees
        model.n_estimators = 20
        model.fit(X, y)
        preds_second = np.array(model.predict(X))

        # Predictions should differ after more rounds
        self.assertFalse(np.allclose(preds_first, preds_second))

    def test_custom_objective_training_policy_auto(self):
        """Custom objective with training_policy='auto'."""
        X, y = _regression_data()
        model = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=20,
            training_policy="auto",
            seed=1,
            deterministic=True,
        )
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))

    def test_custom_objective_training_policy_manual(self):
        """Custom objective with training_policy='manual'."""
        X, y = _regression_data()
        model = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=20,
            training_policy="manual",
            seed=1,
            deterministic=True,
        )
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))


class TestClassifierCustomObjective(unittest.TestCase):
    """GBMClassifier with custom objective."""

    def test_classifier_custom_objective_binary(self):
        """GBMClassifier with custom logistic loss for binary classification."""
        X, y = _classification_data()
        clf = GBMClassifier(
            objective=_custom_logistic_objective,
            n_estimators=20,
            seed=1,
            deterministic=True,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(len(preds), len(y))
        # All predictions should be 0 or 1
        self.assertTrue(all(p in (0, 1) for p in preds))


class TestRankerCustomObjective(unittest.TestCase):
    """GBMRanker with custom objective."""

    def test_ranker_custom_objective_with_group(self):
        """GBMRanker with custom objective + group."""
        X, y, group = _ranking_data()
        rnk = GBMRanker(
            objective=_custom_pairwise_objective,
            n_estimators=20,
            seed=1,
            deterministic=True,
        )
        rnk.fit(X, y, group=group)
        scores = rnk.predict(X)
        self.assertEqual(len(scores), len(y))


class TestCustomObjectiveValidation(unittest.TestCase):
    """Validation and edge cases for custom objectives."""

    def test_non_callable_raises(self):
        """Non-callable objective raises TypeError."""
        with self.assertRaises(TypeError):
            GBMRegressor(objective=42)

    def test_wrong_return_shape_raises(self):
        """Callable returning wrong shape raises error during fit."""
        def bad_obj(y_true, y_pred):
            # Return wrong shape — only 1 element instead of len(y_true)
            return np.array([1.0]), np.array([1.0])

        X, y = _regression_data()
        model = GBMRegressor(objective=bad_obj, n_estimators=5, seed=1)
        with self.assertRaises(Exception):
            model.fit(X, y)

    def test_get_params_set_params_with_callable(self):
        """get_params / set_params work with callable objective."""
        model = GBMRegressor(objective=_custom_mse_objective, seed=1)
        params = model.get_params()
        self.assertIs(params["objective"], _custom_mse_objective)

        # set_params with a different callable
        model.set_params(objective=_custom_logistic_objective)
        self.assertIs(model.objective, _custom_logistic_objective)

        # set_params back to None
        model.set_params(objective=None)
        self.assertIsNone(model.objective)

    def test_repr_with_callable(self):
        """__repr__ works with callable objective."""
        model = GBMRegressor(objective=_custom_mse_objective, seed=1)
        r = repr(model)
        self.assertIn("objective=", r)

    def test_pickle_drops_callable_with_warning(self):
        """Pickle roundtrip drops callable objective with a warning."""
        X, y = _regression_data()
        model = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=10,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y)
        preds_before = model.predict(X[:5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = pickle.dumps(model)
            # Should have emitted a warning about the non-serializable callable
            custom_warnings = [
                x for x in w if "Custom objective callable cannot be pickled" in str(x.message)
            ]
            self.assertGreater(len(custom_warnings), 0)

        model2 = pickle.loads(data)
        self.assertIsNone(model2.objective)
        # Model should still predict correctly
        preds_after = model2.predict(X[:5])
        np.testing.assert_allclose(preds_before, preds_after, rtol=0.0, atol=0.0)

    def test_save_load_model_with_callable(self):
        """save_model / load_model works with custom objective."""
        X, y = _regression_data()
        model = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=10,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y)
        preds_before = model.predict(X[:5])

        with tempfile.NamedTemporaryFile(suffix=".agbm") as f:
            model.save_model(f.name)
            model2 = GBMRegressor.load_model(f.name)

        # Loaded model should have objective=None but still predict
        self.assertIsNone(model2.objective)
        preds_after = model2.predict(X[:5])
        np.testing.assert_allclose(preds_before, preds_after, rtol=0.0, atol=0.0)

    def test_custom_objective_with_custom_metric_and_early_stopping(self):
        """Custom objective + custom metric + early stopping together."""
        X, y = _regression_data(n=200)
        X_val, y_val = _regression_data(n=100, seed=7)

        def custom_mae(y_true, y_pred):
            return ("custom_mae", float(np.mean(np.abs(y_true - y_pred))), False)

        model = GBMRegressor(
            objective=_custom_mse_objective,
            n_estimators=200,
            early_stopping_rounds=10,
            seed=1,
            deterministic=True,
        )
        model.fit(X, y, eval_set=(X_val, y_val), eval_metric=custom_mae)
        self.assertIsNotNone(model.evals_result_)
        self.assertIn("custom_mae", model.evals_result_["validation"])


if __name__ == "__main__":
    unittest.main()
