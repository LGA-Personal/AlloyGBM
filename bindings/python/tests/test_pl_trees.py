"""End-to-end piecewise-linear (PL) tree tests for GBMRegressor, GBMClassifier, and GBMRanker."""

from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _linear_regression_data(n=300, n_features=4, seed=0):
    """Linear target — leaf_model='linear' should help shallowest trees."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    coefs = np.array([1.0, -2.0, 0.5, 1.5], dtype=np.float32)
    y = (X @ coefs + 0.05 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def _toy_binary_data(n=200, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    logits = X @ rng.standard_normal(n_features).astype(np.float32)
    y = (logits > 0).astype(np.int32)
    return X, y


def _toy_multiclass_data(n=300, n_features=4, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    logits = X @ rng.standard_normal((n_features, n_classes)).astype(np.float32)
    y = np.argmax(logits, axis=1).astype(np.int32)
    return X, y


def _toy_ranking_data(n=200, n_features=4, n_groups=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, 5, size=n).astype(np.int32)
    group_sizes = [n // n_groups] * n_groups
    group_sizes[-1] += n - sum(group_sizes)
    group = np.repeat(np.arange(n_groups), group_sizes).astype(np.int32)
    order = np.argsort(group)
    return X[order], y[order].astype(float), group[order]


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestParamValidation:
    def test_valid_constant(self):
        m = GBMRegressor(leaf_model="constant")
        assert m.leaf_model == "constant"

    def test_valid_linear(self):
        m = GBMRegressor(leaf_model="linear")
        assert m.leaf_model == "linear"

    def test_invalid_leaf_model_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            GBMRegressor(leaf_model="polynomial").fit([[1], [2]], [1, 2])

    def test_set_params_updates_leaf_model(self):
        m = GBMRegressor()
        m.set_params(leaf_model="linear")
        assert m.leaf_model == "linear"

    def test_get_params_contains_leaf_model(self):
        m = GBMRegressor(leaf_model="linear")
        params = m.get_params()
        assert "leaf_model" in params
        assert params["leaf_model"] == "linear"

    def test_repr_contains_leaf_model(self):
        m = GBMRegressor(leaf_model="linear")
        r = repr(m)
        assert "leaf_model='linear'" in r


# ---------------------------------------------------------------------------
# Regressor smoke tests
# ---------------------------------------------------------------------------

class TestPLRegressor:
    def test_fits_and_predicts(self):
        X, y = _linear_regression_data()
        m = GBMRegressor(n_estimators=20, max_depth=3, leaf_model="linear", seed=0)
        m.fit(X, y)
        preds = np.asarray(m.predict(X))
        assert preds.shape == (len(y),)
        assert np.isfinite(preds).all()

    def test_linear_leaves_differ_from_constant(self):
        """With depth=1 stumps and a linear target, linear leaves must produce
        meaningfully different predictions from constant leaves."""
        X, y = _linear_regression_data(seed=1)
        # depth=1 (stumps): linear leaves can capture the intra-leaf linear trend
        # while constant leaves must average it away — so predictions will differ.
        mc = GBMRegressor(n_estimators=10, max_depth=1, learning_rate=0.5,
                          leaf_model="constant", seed=0).fit(X, y)
        ml = GBMRegressor(n_estimators=10, max_depth=1, learning_rate=0.5,
                          leaf_model="linear", seed=0).fit(X, y)
        pc = np.asarray(mc.predict(X))
        pl = np.asarray(ml.predict(X))
        max_diff = float(np.abs(pc - pl).max())
        assert max_diff > 1e-4, (
            f"linear leaves should produce meaningfully different predictions "
            f"from constant leaves (max diff={max_diff:.2e})"
        )

    def test_save_load_round_trip(self):
        X, y = _linear_regression_data(seed=2)
        m = GBMRegressor(n_estimators=15, max_depth=3, leaf_model="linear", seed=0).fit(X, y)
        preds_before = np.asarray(m.predict(X))
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            fname = f.name
        try:
            m.save_model(fname)
            m2 = GBMRegressor.load_model(fname)
            preds_after = np.asarray(m2.predict(X))
        finally:
            os.unlink(fname)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-4,
                                   err_msg="save/load round-trip should preserve predictions")

    def test_pickle_round_trip(self):
        X, y = _linear_regression_data(seed=3)
        m = GBMRegressor(n_estimators=15, max_depth=3, leaf_model="linear", seed=0).fit(X, y)
        preds_before = np.asarray(m.predict(X))
        m2 = pickle.loads(pickle.dumps(m))
        preds_after = np.asarray(m2.predict(X))
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-4,
                                   err_msg="pickle round-trip should preserve predictions")

    def test_convergence_on_linear_target_shallow_trees(self):
        """With shallow trees (depth=1), linear leaves should fit a linear target
        faster than constant leaves — i.e. lower RMSE at same round count."""
        X, y = _linear_regression_data(n=400, seed=5)
        n_rounds = 30
        mc = GBMRegressor(n_estimators=n_rounds, max_depth=1, leaf_model="constant",
                          learning_rate=0.3, seed=0).fit(X, y)
        ml = GBMRegressor(n_estimators=n_rounds, max_depth=1, leaf_model="linear",
                          learning_rate=0.3, seed=0).fit(X, y)
        rmse_c = float(((np.asarray(mc.predict(X)) - y) ** 2).mean() ** 0.5)
        rmse_l = float(((np.asarray(ml.predict(X)) - y) ** 2).mean() ** 0.5)
        assert rmse_l < rmse_c, (
            f"linear leaves (RMSE={rmse_l:.4f}) should outperform constant leaves "
            f"(RMSE={rmse_c:.4f}) on a linear target with shallow trees"
        )

    def test_linear_leaves_keep_quality_on_raw_scale_features(self):
        rng = np.random.default_rng(42)
        n = 360
        X = np.zeros((n, 10), dtype=np.float32)
        X[:, 0] = rng.normal(0.0, 1.0, size=n)
        X[:, 1] = rng.normal(0.0, 5000.0, size=n)
        X[:, 8] = rng.normal(0.0, 0.02, size=n)
        X[:, 9] = rng.normal(50.0, 10000.0, size=n)
        for j in range(2, 8):
            X[:, j] = rng.normal(0.0, 1.0 + j, size=n)
        y = (0.6 * X[:, 8] - 0.00008 * X[:, 9] + 0.1 * X[:, 0]).astype(np.float32)

        common_params = dict(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.2,
            lambda_l2=0.01,
            training_policy="manual",
            continuous_binning_strategy="quantile",
            seed=0,
        )
        constant = GBMRegressor(leaf_model="constant", **common_params).fit(X, y)
        linear = GBMRegressor(leaf_model="linear", **common_params).fit(X, y)
        constant_rmse = float(
            np.sqrt(np.mean((np.asarray(constant.predict(X), dtype=np.float32) - y) ** 2))
        )
        linear_predictions = np.asarray(linear.predict(X), dtype=np.float32)
        linear_rmse = float(np.sqrt(np.mean((linear_predictions - y) ** 2)))

        assert np.isfinite(linear_predictions).all()
        assert linear_rmse < 0.20
        assert linear_rmse <= constant_rmse * 0.80, (
            "linear leaves should retain their raw-scale advantage after numeric-solve changes: "
            f"linear RMSE={linear_rmse:.4f}, constant RMSE={constant_rmse:.4f}"
        )


# ---------------------------------------------------------------------------
# Classifier smoke tests
# ---------------------------------------------------------------------------

class TestPLClassifier:
    def test_binary_fits_and_predicts(self):
        X, y = _toy_binary_data()
        clf = GBMClassifier(n_estimators=15, max_depth=3, leaf_model="linear", seed=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})
        proba = np.asarray(clf.predict_proba(X))
        assert proba.shape == (len(y), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_multiclass_fits_and_predicts(self):
        X, y = _toy_multiclass_data()
        clf = GBMClassifier(n_estimators=15, max_depth=3, leaf_model="linear", seed=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1, 2})
        proba = np.asarray(clf.predict_proba(X))
        assert proba.shape == (len(y), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_pickle_round_trip(self):
        X, y = _toy_binary_data(seed=7)
        clf = GBMClassifier(n_estimators=10, max_depth=2, leaf_model="linear", seed=0).fit(X, y)
        preds_before = clf.predict(X)
        clf2 = pickle.loads(pickle.dumps(clf))
        preds_after = clf2.predict(X)
        assert preds_before == preds_after


# ---------------------------------------------------------------------------
# Ranker smoke tests
# ---------------------------------------------------------------------------

class TestPLRanker:
    def test_fits_and_predicts(self):
        X, y, group = _toy_ranking_data()
        rnk = GBMRanker(n_estimators=10, max_depth=3, leaf_model="linear", seed=0)
        rnk.fit(X, y, group=group)
        scores = np.asarray(rnk.predict(X))
        assert scores.shape == (len(y),)
        assert np.isfinite(scores).all()

    def test_save_load_round_trip(self):
        X, y, group = _toy_ranking_data(seed=9)
        rnk = GBMRanker(n_estimators=10, max_depth=3, leaf_model="linear", seed=0)
        rnk.fit(X, y, group=group)
        scores_before = np.asarray(rnk.predict(X))
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            fname = f.name
        try:
            rnk.save_model(fname)
            rnk2 = GBMRanker.load_model(fname)
            scores_after = np.asarray(rnk2.predict(X))
        finally:
            os.unlink(fname)
        np.testing.assert_allclose(scores_before, scores_after, atol=1e-4)


class TestPlTreesShap:
    """SHAP supports `leaf_model='linear'` artifacts as an interventional
    decomposition (path-attribution on the leaf "constant part"
    `intercept + Σ wj·μj_global` plus per-visited-node row deviations
    `wj · (xj − μj_global)`).

    v0.7.3 fixed the SHAP path walker so it uses the same float
    thresholds the predictor uses (`shap::BinningContext` +
    `shap_explain_rows_with_binning`), eliminating the path-walk vs.
    predict-path divergence on continuous features for *scalar*-leaf
    artifacts.

    v0.7.4 closed the linear-leaf gap by walking the row's full path and
    crediting `Σⱼ wⱼ·(xⱼ − μⱼ)` at every visited node — matching how
    `predict` accumulates `leaf.eval_row(row)` at each visited node.
    Strict additivity (`atol + rtol·|predict(x)|`) now holds for
    `leaf_model="linear"` on the default predictor-aligned binning path
    (see `test_shap_pl_strict_additivity.py` for the full coverage
    matrix).  This class only smoke-checks shape and finiteness; the
    strict-additivity contract is enforced in the dedicated test.
    """

    def test_shap_values_return_for_linear_leaf_regressor(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 4)).astype("float32")
        y = X @ np.array([0.5, -0.3, 0.2, 0.1]).astype("float32") + 0.1 * rng.standard_normal(60).astype("float32")
        m = GBMRegressor(n_estimators=5, leaf_model="linear", max_depth=2).fit(X, y)
        ev, shap_values = m.shap_values(X[:3], include_expected_value=True)
        assert np.shape(shap_values) == (3, 4)
        assert np.isfinite(ev)
        assert np.all(np.isfinite(shap_values))
