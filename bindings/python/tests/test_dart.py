"""DART boosting-mode integration tests (v0.9.0).

Covers regressor / binary classifier / ranker smoke, param validation,
multiclass rejection, warm-start rejection, pickle round-trip, and a
sanity check that DART produces different predictions from Standard
(catches a regression where tree_weight is dropped at artifact I/O).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


# ----- Smoke tests -----


def test_dart_regressor_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y = (
        X @ np.array([1.0, -0.5, 0.25, 0.0], dtype=np.float32)
        + 0.1 * rng.normal(size=200).astype(np.float32)
    )
    m = GBMRegressor(
        n_estimators=50,
        boosting_mode="dart",
        dart_drop_rate=0.1,
        dart_max_drop=5,
        seed=42,
    )
    m.fit(X, y)
    preds = np.asarray(m.predict(X))
    assert preds.shape == (200,)
    assert np.isfinite(preds).all()


def test_dart_classifier_binary_predict_proba_in_range():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = GBMClassifier(
        n_estimators=50,
        boosting_mode="dart",
        dart_drop_rate=0.1,
        seed=42,
    )
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (200, 2)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_dart_ranker_fits_with_groups():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    groups = [10] * 10
    m = GBMRanker(
        n_estimators=30,
        boosting_mode="dart",
        dart_drop_rate=0.1,
        ranking_objective="rank:pairwise",
        seed=42,
    )
    m.fit(X, y, group=groups)
    preds = np.asarray(m.predict(X))
    assert preds.shape == (100,)
    assert np.isfinite(preds).all()


# ----- Param-validation tests -----


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dart_drop_rate": -0.1}, "dart_drop_rate"),
        ({"dart_drop_rate": 1.5}, "dart_drop_rate"),
        ({"dart_max_drop": 0}, "dart_max_drop"),
        ({"dart_normalize_type": "bad"}, "dart_normalize_type"),
        ({"dart_sample_type": "bad"}, "dart_sample_type"),
    ],
)
def test_dart_invalid_params_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GBMRegressor(boosting_mode="dart", **kwargs)


# ----- Multiclass rejection -----


def test_multiclass_dart_rejected():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 3)).astype(np.float32)
    y = rng.integers(0, 3, size=60).tolist()  # 3 classes -> softmax path
    m = GBMClassifier(
        n_estimators=10,
        boosting_mode="dart",
        dart_drop_rate=0.1,
    )
    with pytest.raises(NotImplementedError, match=r"multiclass"):
        m.fit(X, y)


# ----- Warm-start rejection -----


def test_dart_warm_start_rejected_at_second_fit():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = rng.normal(size=100).astype(np.float32)
    m = GBMRegressor(
        n_estimators=10,
        boosting_mode="dart",
        dart_drop_rate=0.1,
        warm_start=True,
        seed=42,
    )
    m.fit(X, y)
    # Second fit triggers the warm-start path. The Rust engine
    # rejects DART + warm_start; the Python layer surfaces this as
    # an exception (the specific type depends on whether the engine
    # error reaches us as RuntimeError or ValueError, so accept both).
    with pytest.raises((RuntimeError, ValueError), match=r"warm_start|dart"):
        m.fit(X, y)


# ----- Persistence -----


def test_dart_artifact_round_trips_through_pickle():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(150, 4)).astype(np.float32)
    y = rng.normal(size=150).astype(np.float32)
    m = GBMRegressor(
        n_estimators=30,
        boosting_mode="dart",
        dart_drop_rate=0.1,
        dart_max_drop=5,
        seed=42,
    )
    m.fit(X, y)
    pred_before = np.asarray(m.predict(X))

    blob = pickle.dumps(m)
    m2 = pickle.loads(blob)
    pred_after = np.asarray(m2.predict(X))

    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5, atol=1e-6)


# ----- Numerical sanity: DART != Standard -----


def test_dart_produces_different_model_than_standard():
    """If tree_weight is silently dropped at artifact I/O, predictions
    from a DART-trained model would collapse to the corresponding
    Standard model. This test guards against that regression."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    y = X @ rng.normal(size=5).astype(np.float32) + 0.1 * rng.normal(size=300).astype(
        np.float32
    )

    m_std = GBMRegressor(n_estimators=50, boosting_mode="standard", seed=42)
    m_std.fit(X, y)
    m_dart = GBMRegressor(
        n_estimators=50,
        boosting_mode="dart",
        dart_drop_rate=0.15,
        dart_max_drop=5,
        seed=42,
    )
    m_dart.fit(X, y)

    diff = np.abs(np.asarray(m_std.predict(X)) - np.asarray(m_dart.predict(X))).max()
    assert diff > 1e-3, (
        "DART and Standard should produce different predictions; "
        f"max diff was {diff} (suggests tree_weight wasn't applied)"
    )


def test_dart_forest_normalize_works():
    """Forest-normalize is the alternative to tree-normalize. Make sure
    it doesn't crash and produces finite predictions."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(150, 3)).astype(np.float32)
    y = rng.normal(size=150).astype(np.float32)
    m = GBMRegressor(
        n_estimators=30,
        boosting_mode="dart",
        dart_drop_rate=0.15,
        dart_max_drop=5,
        dart_normalize_type="forest",
        seed=42,
    )
    m.fit(X, y)
    preds = np.asarray(m.predict(X))
    assert np.isfinite(preds).all()
