"""Multiclass softmax + GOSS tests (v0.10.1).

v0.10.0 rejected boosting_mode='goss' for K>=3 classes; v0.10.1 enables
it using per-row scoring s_i = sum_k |g_{i,k}| (LightGBM convention).
"""
import numpy as np
import pytest

from alloygbm import GBMClassifier


def _toy_multiclass(n_rows=200, n_features=5, n_classes=3, seed=21):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_rows).astype(np.int64)
    return X, y


def test_multiclass_goss_trains_and_predicts_proba():
    X, y = _toy_multiclass()
    m = GBMClassifier(
        n_estimators=10,
        boosting_mode="goss",
        goss_top_rate=0.2,
        goss_other_rate=0.1,
        seed=21,
    )
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    labels = m.predict(X)
    assert len(labels) == X.shape[0]


def test_multiclass_goss_pickle_round_trip():
    import pickle

    X, y = _toy_multiclass()
    m = GBMClassifier(
        n_estimators=8, boosting_mode="goss", goss_top_rate=0.2, goss_other_rate=0.1, seed=21,
    )
    m.fit(X, y)
    p1 = m.predict_proba(X)
    restored = pickle.loads(pickle.dumps(m))
    p2 = restored.predict_proba(X)
    np.testing.assert_allclose(p1, p2, rtol=1e-6)


def test_multiclass_goss_different_from_standard():
    """GOSS subsamples rows, so the resulting model differs from a
    Standard fit at the same seed; this is the minimum sanity check
    that the new code path actually ran instead of silently routing to
    Standard."""
    X, y = _toy_multiclass()
    g = GBMClassifier(
        n_estimators=10, boosting_mode="goss", goss_top_rate=0.2, goss_other_rate=0.1, seed=21,
    )
    g.fit(X, y)
    s = GBMClassifier(n_estimators=10, boosting_mode="standard", seed=21)
    s.fit(X, y)
    assert not np.allclose(g.predict_proba(X), s.predict_proba(X), atol=1e-4)
