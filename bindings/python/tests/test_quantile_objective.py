"""Tests for the quantile regression objective."""

from __future__ import annotations

import pickle
import numpy as np
import pytest

from alloygbm import GBMRegressor


def test_quantile_objective_regression() -> None:
    # Set up synthetic data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2)).astype(np.float32)
    # y has a linear component and heterogeneous noise
    y = (2.0 * X[:, 0] + rng.normal(scale=0.5, size=100)).astype(np.float32)

    # Train median regressor (alpha = 0.5)
    model = GBMRegressor(
        objective="quantile",
        quantile_alpha=0.5,
        n_estimators=30,
        learning_rate=0.1,
        max_depth=3,
        training_policy="manual",
        deterministic=True,
        seed=42,
    )
    model.fit(X, y)

    # Sensible predictions check: MAE should be reasonable
    preds = np.asarray(model.predict(X))
    mae = np.mean(np.abs(y - preds))
    assert mae < 0.5, f"MAE too high for median regression: {mae}"


def test_quantile_objective_parameter_validation() -> None:
    # 1. Invalid quantile_alpha in __init__
    for bad_alpha in [-0.5, 0.0, 1.0, 1.5]:
        with pytest.raises(ValueError, match="quantile_alpha"):
            GBMRegressor(objective="quantile", quantile_alpha=bad_alpha)

    # 2. Invalid quantile_alpha type in __init__
    with pytest.raises(Exception):
        GBMRegressor(objective="quantile", quantile_alpha="abc")

    # 3. set_params validation on changing quantile_alpha
    model = GBMRegressor(objective="quantile", quantile_alpha=0.5)
    for bad_alpha in [-0.1, 0.0, 1.0, 1.1]:
        with pytest.raises(ValueError, match="quantile_alpha"):
            model.set_params(quantile_alpha=bad_alpha)

    # 4. set_params validation when changing objective to quantile while quantile_alpha is invalid
    model_bypass = GBMRegressor(objective="squared_error")
    model_bypass.quantile_alpha = 1.5  # Bypass set_params validation
    with pytest.raises(ValueError, match="quantile_alpha"):
        model_bypass.set_params(objective="quantile")


def test_quantile_objective_pickling() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    y = (rng.normal(size=(50, 1)) + rng.normal(scale=0.1, size=(50, 1))).flatten().astype(np.float32)

    model = GBMRegressor(
        objective="quantile",
        quantile_alpha=0.3,
        n_estimators=5,
        training_policy="manual",
        deterministic=True,
        seed=42,
    )
    model.fit(X, y)

    p1 = np.asarray(model.predict(X[:10]))
    blob = pickle.dumps(model)
    restored = pickle.loads(blob)
    p2 = np.asarray(restored.predict(X[:10]))
    np.testing.assert_allclose(p1, p2, rtol=1e-6)
    assert restored.quantile_alpha == 0.3


def test_quantile_objective_evals_result() -> None:
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(80, 2)).astype(np.float32)
    y_train = rng.normal(size=80).astype(np.float32)
    X_val = rng.normal(size=(20, 2)).astype(np.float32)
    y_val = rng.normal(size=20).astype(np.float32)

    model = GBMRegressor(
        objective="quantile",
        quantile_alpha=0.7,
        n_estimators=5,
        training_policy="manual",
        deterministic=True,
        seed=42,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    assert model.evals_result_ is not None
    assert "train" in model.evals_result_
    assert "validation" in model.evals_result_

    # The loss metric name for objective='quantile' should be 'quantile'
    assert "quantile" in model.evals_result_["train"]
    assert "quantile" in model.evals_result_["validation"]

    assert len(model.evals_result_["train"]["quantile"]) == 5
    assert len(model.evals_result_["validation"]["quantile"]) == 5


def test_quantile_empirical_quantile_property() -> None:
    rng = np.random.default_rng(42)
    # Generate enough points so the empirical quantile is stable
    X = rng.uniform(-2, 2, size=(1000, 1)).astype(np.float32)
    # y = X + noise
    y = (X[:, 0] + rng.normal(scale=0.5, size=1000)).astype(np.float32)

    for alpha in [0.1, 0.5, 0.9]:
        model = GBMRegressor(
            objective="quantile",
            quantile_alpha=alpha,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            training_policy="manual",
            deterministic=True,
            seed=42,
        )
        model.fit(X, y)
        preds = np.asarray(model.predict(X))
        
        # Check that y < preds is approximately alpha.
        # Note: For N=1000, the standard error of the empirical alpha-quantile is
        # approx sqrt(alpha * (1 - alpha) / N), which is at most 0.0158 (at alpha=0.5).
        # A tolerance of 0.05 gives a safe ~3-sigma margin to prevent test flakiness.
        underprediction_rate = np.mean(y < preds)
        assert np.abs(underprediction_rate - alpha) < 0.05, (
            f"For alpha={alpha}, empirical underprediction rate is {underprediction_rate}"
        )


def test_quantile_supported_combinations() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2)).astype(np.float32)
    y = (2.0 * X[:, 0] + rng.normal(scale=0.5, size=100)).astype(np.float32)

    # 1. DART + Quantile
    model_dart = GBMRegressor(
        objective="quantile",
        boosting_mode="dart",
        n_estimators=5,
        training_policy="manual",
        deterministic=True,
        seed=42,
    )
    model_dart.fit(X, y)
    preds = model_dart.predict(X)
    assert len(preds) == 100

    # 2. MorphBoost + Quantile
    model_morph = GBMRegressor(
        objective="quantile",
        training_mode="morph",
        n_estimators=5,
        training_policy="manual",
        deterministic=True,
        seed=42,
    )
    model_morph.fit(X, y)
    preds = model_morph.predict(X)
    assert len(preds) == 100

    # 3. Linear Leaves + Quantile
    model_linear = GBMRegressor(
        objective="quantile",
        leaf_model="linear",
        n_estimators=5,
        training_policy="manual",
        deterministic=True,
        seed=42,
    )
    model_linear.fit(X, y)
    preds = model_linear.predict(X)
    assert len(preds) == 100

    # 4. GBMClassifier rejects
    from alloygbm import GBMClassifier
    with pytest.raises(ValueError, match="GBMClassifier does not support objective='quantile'"):
        GBMClassifier(objective="quantile")
    clf = GBMClassifier()
    with pytest.raises(ValueError, match="GBMClassifier does not support objective='quantile'"):
        clf.set_params(objective="quantile")

    # 5. GBMRanker rejects
    from alloygbm import GBMRanker
    with pytest.raises(ValueError, match="GBMRanker does not support objective='quantile'"):
        GBMRanker(objective="quantile")
    ranker = GBMRanker()
    with pytest.raises(ValueError, match="GBMRanker does not support objective='quantile'"):
        ranker.set_params(objective="quantile")

    # 6. MultiLabelGBMRanker rejects
    from alloygbm import MultiLabelGBMRanker
    with pytest.raises(ValueError, match="MultiLabelGBMRanker does not support objective='quantile'"):
        MultiLabelGBMRanker(ranking_objective="quantile")
    with pytest.raises(ValueError, match="MultiLabelGBMRanker does not support objective='quantile'"):
        MultiLabelGBMRanker(objective="quantile")
    mranker = MultiLabelGBMRanker()
    with pytest.raises(ValueError, match="MultiLabelGBMRanker does not support objective='quantile'"):
        mranker.set_params(ranking_objective="quantile")
    with pytest.raises(ValueError, match="MultiLabelGBMRanker does not support objective='quantile'"):
        mranker.set_params(objective="quantile")
