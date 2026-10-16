"""Opt-in leaf math: bounded log-link Hessians and gradient-regularized Newton."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor, MultiLabelGBMRanker


def _count_data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    mu = np.exp(0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3)
    y = rng.poisson(mu).astype(np.float32)
    return X, y


def _positive_data(n: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    mu = np.exp(0.6 * X[:, 0] + 0.4 * X[:, 2])
    y = rng.gamma(shape=2.0, scale=mu / 2.0).astype(np.float32) + 1e-3
    return X, y


def _charbonnier(y_true, y_pred):
    r = y_pred - y_true
    s = np.sqrt(1.0 + r * r)
    return r / s, 1.0 / s**3


def _charbonnier_loss(y_true, y_pred) -> float:
    r = y_pred - y_true
    return float(np.mean(np.sqrt(1.0 + r * r) - 1.0))


def test_defaults_and_params_round_trip():
    model = GBMRegressor()
    params = model.get_params()
    assert params["glm_hessian"] == "exact"
    assert params["grn_strength"] == 0.0
    assert params["grn_form"] == "per_leaf"
    model.set_params(glm_hessian="bounded", grn_strength=0.5, grn_form="per_row")
    assert model.get_params()["glm_hessian"] == "bounded"
    assert model.get_params()["grn_strength"] == 0.5
    assert model.get_params()["grn_form"] == "per_row"
    assert "grn_form='per_row'" in repr(model)
    for cls in (GBMClassifier, GBMRanker):
        assert "glm_hessian" in cls().get_params()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"glm_hessian": "halley"},
        {"grn_strength": -1.0},
        {"grn_strength": float("nan")},
        {"grn_strength": True},
        {"grn_form": "per_tree"},
    ],
)
def test_invalid_values_rejected(kwargs):
    X, y = _count_data(n=50)
    with pytest.raises(ValueError):
        GBMRegressor(n_estimators=2, **kwargs).fit(X, y)
    with pytest.raises(ValueError):
        GBMRegressor(n_estimators=2).set_params(**kwargs).fit(X, y)


@pytest.mark.parametrize("objective", ["poisson", "gamma", "tweedie", "squared_error"])
def test_explicit_defaults_are_byte_identical(objective):
    X, y = _positive_data()
    common = dict(objective=objective, n_estimators=20, learning_rate=0.1, seed=3)
    base = GBMRegressor(**common).fit(X, y)
    explicit = GBMRegressor(
        **common, glm_hessian="exact", grn_strength=0.0, grn_form="per_row"
    ).fit(X, y)
    assert base.artifact_bytes == explicit.artifact_bytes


@pytest.mark.parametrize(
    ("objective", "data"),
    [("poisson", _count_data), ("gamma", _positive_data), ("tweedie", _count_data)],
)
def test_bounded_hessian_changes_log_link_fits(objective, data):
    X, y = data()
    common = dict(objective=objective, n_estimators=30, learning_rate=0.3, seed=0)
    exact = GBMRegressor(**common).fit(X, y)
    bounded = GBMRegressor(**common, glm_hessian="bounded").fit(X, y)
    assert exact.artifact_bytes != bounded.artifact_bytes
    pred = bounded.predict(X)
    assert np.all(np.isfinite(pred)) and np.all(pred > 0.0)
    # Both fits should be in the same ballpark on training data.
    assert np.mean((pred - y) ** 2) < 2.0 * np.mean((exact.predict(X) - y) ** 2) + 1e-6


def test_bounded_hessian_is_inert_for_other_objectives():
    X, y = _positive_data()
    common = dict(objective="squared_error", n_estimators=10, seed=0)
    base = GBMRegressor(**common).fit(X, y)
    bounded = GBMRegressor(**common, glm_hessian="bounded").fit(X, y)
    assert base.artifact_bytes == bounded.artifact_bytes


def test_grn_per_row_rescues_charbonnier_newton_at_learning_rate_one():
    # The paper's Figure 1 setting: plain Newton on Charbonnier loss diverges at
    # learning rate 1.0 because the Hessian vanishes for large residuals.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 5))
    y = 10 * X[:, 0] + 5 * np.sin(2 * X[:, 1]) + 20 + 0.5 * rng.normal(size=1000)
    common = dict(
        objective=_charbonnier,
        n_estimators=40,
        learning_rate=1.0,
        max_depth=4,
        training_policy="manual",
        seed=0,
    )
    newton = GBMRegressor(**common).fit(X, y)
    grn = GBMRegressor(**common, grn_strength=1.0, grn_form="per_row").fit(X, y)
    newton_loss = _charbonnier_loss(y, newton.predict(X))
    grn_loss = _charbonnier_loss(y, grn.predict(X))
    assert grn_loss < 1.0
    assert newton_loss > 100 * grn_loss


@pytest.mark.parametrize("form", ["per_leaf", "per_row"])
def test_grn_lambda_reported_per_round(form):
    X, y = _count_data()
    model = GBMRegressor(
        objective="poisson", n_estimators=10, grn_strength=0.5, grn_form=form, seed=0
    ).fit(X, y)
    lambdas = [d["grn_lambda"] for d in model.diagnostics_per_round_]
    assert len(lambdas) == len(model.diagnostics_per_round_) > 0
    assert all(lam is not None and lam > 0.0 for lam in lambdas)

    off = GBMRegressor(objective="poisson", n_estimators=5, seed=0).fit(X, y)
    assert all(d["grn_lambda"] is None for d in off.diagnostics_per_round_)


def test_grn_multiclass_fits():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5).astype(int)
    base = GBMClassifier(n_estimators=10, seed=0).fit(X, y)
    grn = GBMClassifier(n_estimators=10, seed=0, grn_strength=1.0).fit(X, y)
    assert base.artifact_bytes != grn.artifact_bytes
    proba = grn.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert all(d["grn_lambda"] > 0.0 for d in grn.diagnostics_per_round_)


def test_joint_mode_rejects_leaf_math_options():
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = np.random.default_rng(1).integers(0, 2, size=(40, 2))
    group = np.repeat(np.arange(8), 5)
    for kwargs in ({"grn_strength": 1.0}, {"glm_hessian": "bounded"}):
        model = MultiLabelGBMRanker(multi_label_mode="joint", n_estimators=3, **kwargs)
        with pytest.raises(NotImplementedError):
            model.fit(X, y, group=group)


def test_pickle_from_before_the_options_existed():
    X, y = _count_data(n=100)
    model = GBMRegressor(objective="poisson", n_estimators=3).fit(X, y)
    state = model.__getstate__()
    for key in ("glm_hessian", "grn_strength", "grn_form"):
        state.pop(key, None)
    restored = GBMRegressor.__new__(GBMRegressor)
    restored.__setstate__(state)
    assert restored.get_params()["glm_hessian"] == "exact"
    assert restored.get_params()["grn_strength"] == 0.0
    assert restored.get_params()["grn_form"] == "per_leaf"
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    pickle.loads(pickle.dumps(restored))
