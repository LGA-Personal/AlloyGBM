"""End-to-end morph mode tests for GBMRegressor, GBMClassifier, and GBMRanker."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


def _toy_regression_data(n=200, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    coefs = rng.standard_normal(n_features).astype(np.float32)
    y = X @ coefs + 0.1 * rng.standard_normal(n).astype(np.float32)
    return X, y


def _toy_binary_data(n=200, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    logits = X @ rng.standard_normal(n_features).astype(np.float32)
    y = (logits > 0).astype(np.int32)
    return X, y


def _toy_multiclass_data(n=300, n_features=5, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    logits = X @ rng.standard_normal((n_features, n_classes)).astype(np.float32)
    y = np.argmax(logits, axis=1).astype(np.int32)
    return X, y


def _toy_ranking_data(n=200, n_features=5, n_groups=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, 5, size=n).astype(np.int32)
    group_sizes = [n // n_groups] * n_groups
    group_sizes[-1] += n - sum(group_sizes)
    group = np.repeat(np.arange(n_groups), group_sizes).astype(np.int32)
    order = np.argsort(group)
    return X[order], y[order].astype(float), group[order]


# --- Regressor smoke tests ---

def test_regressor_fits_in_morph_mode():
    X, y = _toy_regression_data()
    m = GBMRegressor(n_estimators=20, max_depth=4, learning_rate=0.1,
                     training_mode="morph", seed=42)
    m.fit(X, y)
    pred = np.asarray(m.predict(X))
    assert pred.shape == (len(y),)
    assert np.isfinite(pred).all()


def test_regressor_morph_mode_round_trips_via_pickle():
    X, y = _toy_regression_data()
    m = GBMRegressor(n_estimators=10, max_depth=3, training_mode="morph", seed=0)
    m.fit(X, y)
    blob = pickle.dumps(m)
    m2 = pickle.loads(blob)
    np.testing.assert_array_equal(m.predict(X), m2.predict(X))


def test_regressor_morph_mode_introspection():
    import inspect
    sig = inspect.signature(GBMRegressor.__init__)
    for p in ("training_mode", "morph_rate", "evolution_pressure",
              "morph_warmup_iters", "lr_schedule", "lr_warmup_frac"):
        assert p in sig.parameters, f"missing param: {p}"


def test_regressor_get_set_params_round_trip_morph():
    m = GBMRegressor(n_estimators=5, training_mode="morph", morph_rate=0.2,
                     lr_schedule="warmup_cosine", lr_warmup_frac=0.15)
    p = m.get_params()
    assert p["training_mode"] == "morph"
    assert p["morph_rate"] == 0.2
    assert p["lr_schedule"] == "warmup_cosine"
    m2 = GBMRegressor()
    m2.set_params(**p)
    assert m2.training_mode == "morph"
    assert m2.lr_schedule == "warmup_cosine"


def test_regressor_default_mode_unchanged():
    X, y = _toy_regression_data()
    m1 = GBMRegressor(n_estimators=10, max_depth=4, seed=0)
    m1.fit(X, y)
    m2 = GBMRegressor(n_estimators=10, max_depth=4, seed=0)
    m2.fit(X, y)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


# --- Classifier smoke tests ---

def test_classifier_fits_in_morph_mode_binary():
    X, y = _toy_binary_data()
    m = GBMClassifier(n_estimators=15, max_depth=4, training_mode="morph", seed=0)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classifier_fits_in_morph_mode_multiclass():
    X, y = _toy_multiclass_data()
    m = GBMClassifier(n_estimators=15, max_depth=4, training_mode="morph", seed=0)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# --- Ranker smoke tests ---

def test_ranker_fits_in_morph_mode():
    X, y, group = _toy_ranking_data()
    m = GBMRanker(n_estimators=10, max_depth=4, training_mode="morph", seed=0)
    m.fit(X, y, group=group)
    pred = np.asarray(m.predict(X))
    assert pred.shape == (len(y),)
    assert np.isfinite(pred).all()


def test_ranker_init_signature_exposes_morph_params():
    import inspect
    sig = inspect.signature(GBMRanker.__init__)
    assert "training_mode" in sig.parameters
    assert "morph_rate" in sig.parameters


# --- Determinism + backward compat (Task 11) ---

def test_morph_artifact_bytes_identical_for_same_seed():
    X, y = _toy_regression_data(n=300, seed=7)
    kw = dict(n_estimators=15, max_depth=4, learning_rate=0.1,
              training_mode="morph", seed=12345)
    m1 = GBMRegressor(**kw)
    m1.fit(X, y)
    m2 = GBMRegressor(**kw)
    m2.fit(X, y)
    assert m1.artifact_bytes == m2.artifact_bytes


def test_morph_warmup_cosine_artifact_deterministic():
    X, y = _toy_regression_data(n=200, seed=3)
    kw = dict(n_estimators=20, max_depth=4, training_mode="morph",
              lr_schedule="warmup_cosine", lr_warmup_frac=0.2, seed=99)
    m1 = GBMRegressor(**kw)
    m1.fit(X, y)
    m2 = GBMRegressor(**kw)
    m2.fit(X, y)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


def test_auto_mode_artifact_deterministic_after_morph_pr():
    X, y = _toy_regression_data(n=200, seed=11)
    kw = dict(n_estimators=10, max_depth=4, seed=0)
    m1 = GBMRegressor(**kw)
    m1.fit(X, y)
    m2 = GBMRegressor(**kw)
    m2.fit(X, y)
    assert m1.artifact_bytes == m2.artifact_bytes


def test_morph_artifact_differs_from_auto():
    X, y = _toy_regression_data(n=200, seed=13)
    m_auto = GBMRegressor(n_estimators=15, max_depth=4, seed=0)
    m_auto.fit(X, y)
    m_morph = GBMRegressor(n_estimators=15, max_depth=4, training_mode="morph", seed=0)
    m_morph.fit(X, y)
    assert m_auto.artifact_bytes != m_morph.artifact_bytes
