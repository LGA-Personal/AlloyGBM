"""GLM objective smoke tests for v0.11.0: Poisson, Gamma, Tweedie."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMRegressor
from alloygbm.evaluation import gamma_deviance, poisson_deviance, tweedie_deviance


# --- Poisson ----------------------------------------------------------------


def test_poisson_objective_trains_and_predicts_positive() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 4)).astype(np.float32)
    lam = np.exp(0.3 * X[:, 0] + 0.1 * X[:, 1])
    y = rng.poisson(lam).astype(np.float32)
    model = GBMRegressor(
        objective="poisson",
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
        training_policy="manual",
        deterministic=True,
        seed=7,
    )
    model.fit(X, y)
    preds = np.asarray(model.predict(X))
    assert np.all(preds > 0.0), "Poisson predictions must be strictly positive"


def test_poisson_rejects_negative_targets() -> None:
    X = np.zeros((10, 2), dtype=np.float32)
    y = np.array(
        [1.0, -1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
    )
    model = GBMRegressor(objective="poisson", n_estimators=3, training_policy="manual")
    with pytest.raises(Exception) as exc:
        model.fit(X, y)
    msg = str(exc.value).lower()
    assert "non-negative" in msg or "negative" in msg


def test_poisson_deviance_improves_with_training() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    lam = np.exp(0.5 * X[:, 0])
    y = rng.poisson(lam).astype(np.float32)
    baseline_pred = np.full_like(y, y.mean())
    baseline_dev = poisson_deviance(y.tolist(), baseline_pred.tolist())

    model = GBMRegressor(
        objective="poisson",
        n_estimators=50,
        learning_rate=0.05,
        training_policy="manual",
        deterministic=True,
        seed=11,
    )
    model.fit(X, y)
    preds = np.asarray(model.predict(X))
    trained_dev = poisson_deviance(y.tolist(), preds.tolist())
    assert trained_dev < baseline_dev, (
        f"poisson_deviance should improve: baseline={baseline_dev}, trained={trained_dev}"
    )


# --- Gamma ------------------------------------------------------------------


def test_gamma_objective_trains_and_predicts_positive() -> None:
    rng = np.random.default_rng(13)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    mu = np.exp(0.5 + 0.3 * X[:, 0])
    y = rng.gamma(shape=2.0, scale=mu / 2.0).astype(np.float32)
    model = GBMRegressor(
        objective="gamma",
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
        training_policy="manual",
        deterministic=True,
        seed=13,
    )
    model.fit(X, y)
    preds = np.asarray(model.predict(X))
    assert np.all(preds > 0.0)


def test_gamma_rejects_zero_or_negative_targets() -> None:
    X = np.zeros((6, 2), dtype=np.float32)
    y = np.array([1.0, 2.0, 0.0, 3.0, 1.5, 2.5], dtype=np.float32)  # zero present
    model = GBMRegressor(objective="gamma", n_estimators=3, training_policy="manual")
    with pytest.raises(Exception) as exc:
        model.fit(X, y)
    msg = str(exc.value).lower()
    assert "positive" in msg or "> 0" in msg


def test_gamma_deviance_improves_with_training() -> None:
    rng = np.random.default_rng(17)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    mu = np.exp(0.3 + 0.2 * X[:, 0])
    y = rng.gamma(shape=2.0, scale=mu / 2.0).astype(np.float32)
    baseline = np.full_like(y, y.mean())
    baseline_dev = gamma_deviance(y.tolist(), baseline.tolist())

    model = GBMRegressor(
        objective="gamma",
        n_estimators=50,
        learning_rate=0.05,
        training_policy="manual",
        deterministic=True,
        seed=17,
    )
    model.fit(X, y)
    preds = np.asarray(model.predict(X))
    trained_dev = gamma_deviance(y.tolist(), preds.tolist())
    assert trained_dev < baseline_dev


# --- Tweedie ----------------------------------------------------------------


def test_tweedie_objective_trains_and_predicts_positive() -> None:
    rng = np.random.default_rng(19)
    X = rng.normal(size=(150, 3)).astype(np.float32)
    mu = np.exp(0.5 + 0.4 * X[:, 0])
    y = np.where(rng.random(150) < 0.4, 0.0, rng.gamma(2.0, mu / 2.0)).astype(np.float32)
    model = GBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
        training_policy="manual",
        deterministic=True,
        seed=19,
    )
    model.fit(X, y)
    preds = np.asarray(model.predict(X))
    assert np.all(preds > 0.0)


def test_tweedie_rejects_invalid_variance_power() -> None:
    for bad_p in (0.5, 1.0, 2.0, 2.5):
        with pytest.raises(Exception):
            GBMRegressor(
                objective="tweedie",
                tweedie_variance_power=bad_p,
                n_estimators=3,
            )


def test_tweedie_deviance_improves_with_training() -> None:
    rng = np.random.default_rng(23)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    mu = np.exp(0.3 + 0.2 * X[:, 0])
    y = np.where(rng.random(200) < 0.3, 0.0, rng.gamma(2.0, mu / 2.0)).astype(np.float32)
    baseline = np.full_like(y, max(y.mean(), 1e-3))
    baseline_dev = tweedie_deviance(y.tolist(), baseline.tolist(), variance_power=1.5)

    model = GBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=50,
        learning_rate=0.05,
        training_policy="manual",
        deterministic=True,
        seed=23,
    )
    model.fit(X, y)
    preds = np.asarray(model.predict(X))
    trained_dev = tweedie_deviance(y.tolist(), preds.tolist(), variance_power=1.5)
    assert trained_dev < baseline_dev


def test_poisson_model_round_trips_through_pickle() -> None:
    rng = np.random.default_rng(31)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    y = rng.poisson(np.exp(0.3 * X[:, 0])).astype(np.float32)
    model = GBMRegressor(
        objective="poisson",
        n_estimators=5,
        training_policy="manual",
        deterministic=True,
        seed=31,
    )
    model.fit(X, y)
    p1 = np.asarray(model.predict(X[:10]))
    blob = pickle.dumps(model)
    restored = pickle.loads(blob)
    p2 = np.asarray(restored.predict(X[:10]))
    np.testing.assert_allclose(p1, p2, rtol=1e-6)


def test_tweedie_with_dart_smoke() -> None:
    """DART boosting on Tweedie: feature-agnostic gradient handling should just work."""
    rng = np.random.default_rng(37)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    mu = np.exp(0.3 * X[:, 0])
    y = np.where(rng.random(120) < 0.3, 0.0, rng.gamma(2.0, mu / 2.0)).astype(np.float32)
    model = GBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        boosting_mode="dart",
        dart_drop_rate=0.1,
        n_estimators=15,
        training_policy="manual",
        deterministic=True,
        seed=37,
    )
    model.fit(X, y)
    preds = np.asarray(model.predict(X))
    assert np.all(preds > 0.0)
