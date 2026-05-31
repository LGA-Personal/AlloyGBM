"""SHAP interaction values (v0.11.0): public API + invariants."""

from __future__ import annotations

import numpy as np
import pytest

from alloygbm import GBMRegressor


def _trained_regressor(seed: int = 7) -> tuple[GBMRegressor, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + 0.3 * X[:, 2] + rng.normal(scale=0.1, size=80)).astype(
        np.float32
    )
    model = GBMRegressor(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        training_policy="manual",
        deterministic=True,
        seed=seed,
    )
    model.fit(X, y)
    return model, X


def test_shap_interaction_values_returns_correct_shape() -> None:
    model, X = _trained_regressor()
    interactions = model.shap_interaction_values(X[:5])
    arr = np.asarray(interactions)
    assert arr.shape == (5, 3, 3)


def test_shap_interaction_values_additive_to_prediction() -> None:
    model, X = _trained_regressor()
    expected, interactions = model.shap_interaction_values(
        X[:5], include_expected_value=True
    )
    arr = np.asarray(interactions, dtype=np.float64)
    predicted = np.asarray(model.predict(X[:5]), dtype=np.float64)
    reconstructed = arr.sum(axis=(1, 2)) + expected
    np.testing.assert_allclose(reconstructed, predicted, atol=1e-3, rtol=1e-3)


def test_shap_interaction_values_symmetric() -> None:
    model, X = _trained_regressor()
    interactions = np.asarray(model.shap_interaction_values(X[:5]))
    for k in range(interactions.shape[0]):
        np.testing.assert_allclose(interactions[k], interactions[k].T, atol=1e-5)


def test_shap_interaction_values_row_marginal_matches_shap_values() -> None:
    model, X = _trained_regressor()
    interactions = np.asarray(model.shap_interaction_values(X[:5]))
    shap = np.asarray(model.shap_values(X[:5]))
    np.testing.assert_allclose(
        interactions.sum(axis=2), shap, atol=1e-3, rtol=1e-3
    )


def test_shap_interaction_values_accepts_linear_leaf_model() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 3)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    model = GBMRegressor(
        n_estimators=5,
        max_depth=2,
        leaf_model="linear",
        training_policy="manual",
        deterministic=True,
        seed=11,
    )
    model.fit(X, y)
    
    expected, phi_list = model.shap_interaction_values(X, include_expected_value=True)
    phi = np.array(phi_list)
    assert phi.shape == (60, 3, 3)
    
    # Check full additivity
    predicted = model.predict(X)
    reconstructed = phi.sum(axis=(1, 2)) + expected
    np.testing.assert_allclose(reconstructed, predicted, atol=1e-3, rtol=1e-3)


def test_shap_interaction_values_handles_unfit_model() -> None:
    model = GBMRegressor(n_estimators=3)
    with pytest.raises(RuntimeError):
        model.shap_interaction_values(np.zeros((2, 3), dtype=np.float32))


try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


@pytest.mark.skipif(not HAVE_SKLEARN, reason="sklearn not installed")
def test_shap_interactions_additivity_on_california_housing() -> None:
    """Real-data smoke: 100 rows, 8 features, 20 trees, depth 4."""
    data = fetch_california_housing(as_frame=False)
    X_train, X_test, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.1, random_state=11
    )
    model = GBMRegressor(
        n_estimators=20,
        max_depth=4,
        training_policy="manual",
        deterministic=True,
        seed=11,
    )
    model.fit(X_train, y_train)
    rows = X_test[:100]
    expected, interactions = model.shap_interaction_values(
        rows, include_expected_value=True
    )
    arr = np.asarray(interactions, dtype=np.float64)
    predicted = np.asarray(model.predict(rows), dtype=np.float64)
    reconstructed = arr.sum(axis=(1, 2)) + expected
    abs_tol = 1e-5 + 1e-4 * np.abs(predicted)
    assert np.all(np.abs(reconstructed - predicted) < abs_tol + 1e-3), (
        "California Housing additivity violated"
    )
