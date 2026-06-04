"""GLM and Quantile objective tests for GBMRanker and MultiLabelGBMRanker."""

from __future__ import annotations
import numpy as np
import pytest
from alloygbm import GBMRanker, MultiLabelGBMRanker

def test_ranker_glm_objectives() -> None:
    # 1. Poisson GBMRanker
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    lam = np.exp(0.3 * X[:, 0] + 0.1 * X[:, 1])
    y = rng.poisson(lam).astype(np.float32)
    groups = np.array([0]*100 + [1]*100, dtype=np.int32)

    ranker_poisson = GBMRanker(
        ranking_objective="poisson",
        n_estimators=10,
        learning_rate=0.1,
        seed=42,
        deterministic=True,
    )
    ranker_poisson.fit(X, y, group=groups)
    preds_poisson = np.asarray(ranker_poisson.predict(X))
    assert np.all(preds_poisson > 0.0), "Poisson predictions must be positive"

    # 2. Gamma GBMRanker
    y_gamma = rng.gamma(shape=2.0, scale=lam / 2.0).astype(np.float32) + 0.1
    ranker_gamma = GBMRanker(
        ranking_objective="gamma",
        n_estimators=10,
        learning_rate=0.1,
        seed=42,
        deterministic=True,
    )
    ranker_gamma.fit(X, y_gamma, group=groups)
    preds_gamma = np.asarray(ranker_gamma.predict(X))
    assert np.all(preds_gamma > 0.0), "Gamma predictions must be positive"

    # 3. Tweedie GBMRanker
    y_tweedie = np.where(rng.random(200) < 0.2, 0.0, y_gamma).astype(np.float32)
    ranker_tweedie = GBMRanker(
        ranking_objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=10,
        learning_rate=0.1,
        seed=42,
        deterministic=True,
    )
    ranker_tweedie.fit(X, y_tweedie, group=groups)
    preds_tweedie = np.asarray(ranker_tweedie.predict(X))
    assert np.all(preds_tweedie > 0.0), "Tweedie predictions must be positive"


def test_multilabel_ranker_glm_quantile_independent() -> None:
    rng = np.random.default_rng(43)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    groups = np.array([0]*100 + [1]*100, dtype=np.int32)

    # 2-label targets
    y1 = rng.poisson(np.exp(0.2 * X[:, 0])).astype(np.float32)
    y2 = (rng.gamma(shape=2.0, scale=1.0, size=200).astype(np.float32) + 0.1)
    Y = np.column_stack([y1, y2])

    # Independent MultiLabelGBMRanker with mixed or identical objectives
    # Poisson / Gamma
    mranker = MultiLabelGBMRanker(
        ranking_objective=["poisson", "gamma"],
        multi_label_mode="independent",
        n_estimators=10,
        seed=43,
    )
    mranker.fit(X, Y, group=groups)
    preds = np.asarray(mranker.predict(X))
    assert preds.shape == (200, 2)
    assert np.all(preds > 0.0)

    # Quantile / Tweedie
    Y_qt = np.column_stack([y1, np.where(rng.random(200) < 0.2, 0.0, y2)])
    mranker_qt = MultiLabelGBMRanker(
        ranking_objective=["quantile", "tweedie"],
        quantile_alpha=0.3,
        tweedie_variance_power=1.5,
        multi_label_mode="independent",
        n_estimators=10,
        seed=43,
    )
    mranker_qt.fit(X, Y_qt, group=groups)
    preds_qt = np.asarray(mranker_qt.predict(X))
    assert preds_qt.shape == (200, 2)
    assert np.all(preds_qt[:, 1] > 0.0) # Tweedie predictions strictly positive


def test_multilabel_ranker_glm_quantile_joint() -> None:
    rng = np.random.default_rng(44)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    groups = np.array([0]*100 + [1]*100, dtype=np.int32)

    # Poisson / Gamma / Tweedie / Quantile in Joint mode
    y1 = rng.poisson(np.exp(0.2 * X[:, 0])).astype(np.float32)
    y2 = (rng.gamma(shape=2.0, scale=1.0, size=200).astype(np.float32) + 0.1)
    y3 = np.where(rng.random(200) < 0.2, 0.0, y2).astype(np.float32)
    y4 = rng.normal(size=200).astype(np.float32)
    Y = np.column_stack([y1, y2, y3, y4])

    mranker = MultiLabelGBMRanker(
        ranking_objective=["poisson", "gamma", "tweedie", "quantile"],
        quantile_alpha=0.7,
        tweedie_variance_power=1.6,
        multi_label_mode="joint",
        n_estimators=10,
        seed=44,
    )
    mranker.fit(X, Y, group=groups)
    preds = np.asarray(mranker.predict(X))
    assert preds.shape == (200, 4)
    assert np.all(preds[:, 0] > 0.0)
    assert np.all(preds[:, 1] > 0.0)
    assert np.all(preds[:, 2] > 0.0)


def test_multilabel_ranker_glm_save_load_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(45)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    groups = np.array([0]*50 + [1]*50, dtype=np.int32)
    y1 = rng.poisson(np.exp(0.2 * X[:, 0])).astype(np.float32)
    y2 = (rng.gamma(shape=2.0, scale=1.0, size=100).astype(np.float32) + 0.1)
    Y = np.column_stack([y1, y2])

    mranker = MultiLabelGBMRanker(
        ranking_objective=["poisson", "gamma"],
        multi_label_mode="joint",
        n_estimators=5,
        seed=45,
    )
    mranker.fit(X, Y, group=groups)
    preds_before = np.asarray(mranker.predict(X))

    path = tmp_path / "model.alloy"
    mranker.save_model(str(path))
    restored = MultiLabelGBMRanker.load_model(str(path))
    preds_after = np.asarray(restored.predict(X))

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6)
    assert restored.ranking_objective == ["poisson", "gamma"]
    # Check that predictions are indeed post-transformed (positive)
    assert np.all(preds_after > 0.0)

