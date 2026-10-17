"""Opt-in quantized-gradient histograms (``quantized_grad``)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor, MultiLabelGBMRanker


def _regression_data(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = 2.0 * X[:, 0] + np.sin(3.0 * X[:, 1]) + X[:, 2] * X[:, 3] + 0.3 * rng.normal(size=n)
    return X, y.astype(np.float32)


def test_defaults_and_params_round_trip():
    model = GBMRegressor()
    params = model.get_params()
    assert params["quantized_grad"] is False
    assert params["quantized_grad_bits"] == 4
    model.set_params(quantized_grad=True, quantized_grad_bits=3)
    assert model.get_params()["quantized_grad"] is True
    assert model.get_params()["quantized_grad_bits"] == 3
    assert "quantized_grad=True" in repr(model)
    assert "quantized_grad_bits=3" in repr(model)
    for cls in (GBMClassifier, GBMRanker):
        assert "quantized_grad" in cls().get_params()
        assert "quantized_grad" in repr(cls())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantized_grad": 1},
        {"quantized_grad": "yes"},
        {"quantized_grad_bits": 1},
        {"quantized_grad_bits": 9},
        {"quantized_grad_bits": 4.0},
        {"quantized_grad_bits": True},
    ],
)
def test_invalid_values_rejected(kwargs):
    X, y = _regression_data(n=50)
    with pytest.raises(ValueError):
        GBMRegressor(n_estimators=2, **kwargs).fit(X, y)
    with pytest.raises(ValueError):
        GBMRegressor(n_estimators=2).set_params(**kwargs).fit(X, y)


def test_off_is_byte_identical_whatever_the_bit_width():
    X, y = _regression_data(n=500)
    common = dict(n_estimators=20, learning_rate=0.1, seed=3)
    base = GBMRegressor(**common).fit(X, y)
    explicit = GBMRegressor(**common, quantized_grad=False, quantized_grad_bits=7).fit(X, y)
    assert base.artifact_bytes == explicit.artifact_bytes


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_regression_is_reproducible_and_close_to_float(bits):
    X, y = _regression_data()
    X_test, y_test = _regression_data(n=1000, seed=1)
    common = dict(n_estimators=100, learning_rate=0.1, max_depth=5, seed=0)
    float_model = GBMRegressor(**common).fit(X, y)
    quantized = GBMRegressor(**common, quantized_grad=True, quantized_grad_bits=bits).fit(X, y)
    again = GBMRegressor(**common, quantized_grad=True, quantized_grad_bits=bits).fit(X, y)
    assert quantized.artifact_bytes != float_model.artifact_bytes
    assert quantized.artifact_bytes == again.artifact_bytes

    def rmse(model):
        return float(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))

    assert rmse(quantized) < 1.15 * rmse(float_model)


def test_binary_and_ranking_fit():
    X, y = _regression_data()
    labels = (y > np.median(y)).astype(int)
    classifier = GBMClassifier(n_estimators=40, quantized_grad=True, seed=0).fit(X, labels)
    assert np.mean(classifier.predict(X) == labels) > 0.85

    group = np.repeat(np.arange(len(y) // 20), 20)
    relevance = np.digitize(y, np.quantile(y, [0.5, 0.8, 0.95]))
    ranker = GBMRanker(n_estimators=20, quantized_grad=True, seed=0)
    ranker.fit(X, relevance, group=group)
    assert np.all(np.isfinite(ranker.predict(X)))


@pytest.mark.parametrize("strategy", ["per_class_trees", "vector_leaves"])
def test_multiclass_fits_with_either_tree_layout(strategy):
    X, y = _regression_data()
    labels = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))
    common = dict(n_estimators=30, multiclass_strategy=strategy, seed=0)
    base = GBMClassifier(**common).fit(X, labels)
    quantized = GBMClassifier(**common, quantized_grad=True).fit(X, labels)
    assert base.artifact_bytes != quantized.artifact_bytes
    proba = quantized.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    accuracy = np.mean(quantized.predict(X) == labels)
    assert accuracy > np.mean(base.predict(X) == labels) - 0.05


@pytest.mark.parametrize(
    "kwargs",
    [
        {"leaf_solver": "dro", "dro_radius": 0.1},
        {"leaf_model": "linear"},
        {"training_mode": "morph"},
    ],
)
def test_unsupported_modes_rejected(kwargs):
    X, y = _regression_data(n=200)
    with pytest.raises(ValueError, match="quantized gradients"):
        GBMRegressor(n_estimators=3, quantized_grad=True, **kwargs).fit(X, y)


def test_joint_mode_rejects_quantized_grad():
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = np.random.default_rng(1).integers(0, 2, size=(40, 2))
    group = np.repeat(np.arange(8), 5)
    model = MultiLabelGBMRanker(multi_label_mode="joint", n_estimators=3, quantized_grad=True)
    with pytest.raises(NotImplementedError):
        model.fit(X, y, group=group)


def test_pickle_from_before_the_option_existed():
    X, y = _regression_data(n=100)
    model = GBMRegressor(n_estimators=3).fit(X, y)
    state = model.__getstate__()
    for key in ("quantized_grad", "quantized_grad_bits"):
        state.pop(key, None)
    restored = GBMRegressor.__new__(GBMRegressor)
    restored.__setstate__(state)
    assert restored.get_params()["quantized_grad"] is False
    assert restored.get_params()["quantized_grad_bits"] == 4
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    pickle.loads(pickle.dumps(restored))
