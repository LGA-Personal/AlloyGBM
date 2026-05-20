"""Multiclass softmax + DART tests (v0.10.1).

v0.9.0 rejected DART for K>=3 classes; v0.10.0 still deferred it;
v0.10.1 enables it.  DART maintains a flat per-tree `tree_weight`
pool across the K-stumps-per-round commit order; before each round
it picks tree indices to drop (LightGBM convention: drop entire
class-trees, not gradient channels) and after building K new trees
it calls `apply_normalization` to rescale.
"""
import numpy as np
import pytest

from alloygbm import GBMClassifier


def _toy_multiclass(n_rows=200, n_features=5, n_classes=3, seed=23):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_rows).astype(np.int64)
    return X, y


def test_multiclass_dart_trains_and_predicts_proba():
    X, y = _toy_multiclass()
    m = GBMClassifier(
        n_estimators=10,
        boosting_mode="dart",
        dart_drop_rate=0.2,
        dart_max_drop=5,
        seed=23,
    )
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_multiclass_dart_differs_from_standard():
    X, y = _toy_multiclass()
    d = GBMClassifier(
        n_estimators=10, boosting_mode="dart", dart_drop_rate=0.2, seed=23,
    )
    d.fit(X, y)
    s = GBMClassifier(n_estimators=10, boosting_mode="standard", seed=23)
    s.fit(X, y)
    assert not np.allclose(d.predict_proba(X), s.predict_proba(X), atol=1e-4)


def test_multiclass_dart_pickle_round_trip():
    import pickle

    X, y = _toy_multiclass()
    m = GBMClassifier(
        n_estimators=8, boosting_mode="dart", dart_drop_rate=0.2, seed=23,
    )
    m.fit(X, y)
    p1 = m.predict_proba(X)
    restored = pickle.loads(pickle.dumps(m))
    p2 = restored.predict_proba(X)
    np.testing.assert_allclose(p1, p2, rtol=1e-6)


def test_multiclass_dart_warm_start_continues_without_error():
    X, y = _toy_multiclass()
    base = GBMClassifier(
        n_estimators=8, boosting_mode="dart", dart_drop_rate=0.15, seed=23,
    )
    base.fit(X, y)
    cont = GBMClassifier(
        n_estimators=8,
        boosting_mode="dart",
        dart_drop_rate=0.15,
        warm_start=True,
        seed=23,
    )
    cont.fit(X, y, init_model=base)
    p_base = base.predict_proba(X)
    p_cont = cont.predict_proba(X)
    # Continuation should produce a different, valid distribution.
    assert not np.allclose(p_base, p_cont, atol=1e-5)
    assert np.allclose(p_cont.sum(axis=1), 1.0, atol=1e-5)


def test_multiclass_dart_first_round_no_dropouts():
    """select_dropouts returns empty when tree_weights pool is empty,
    so round 0 of a fresh multiclass DART fit must add K new stumps
    with weight 1.0 and no normalization-driven changes."""
    X, y = _toy_multiclass(n_rows=50, seed=31)
    m = GBMClassifier(
        n_estimators=1, boosting_mode="dart", dart_drop_rate=0.5, seed=31,
    )
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (50, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
