"""Greedy quantile border selection (LightGBM ``GreedyFindBin`` style)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from alloygbm import GBMRegressor

EQUAL_FREQUENCY_ENV = "ALLOYGBM_EXPERIMENT_EQUAL_FREQUENCY_BINS"


def _zipf_fixture(seed: int = 7, n: int = 6_000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = rng.zipf(1.6, n).clip(1, 400).astype(np.float32)
    noise = rng.normal(size=n).astype(np.float32)
    x = np.column_stack([k, noise]).astype(np.float32)
    y = (1.5 * np.sin(0.7 * k) + noise).astype(np.float32)
    return x, y


def _cuts(model: GBMRegressor) -> list[list[float]]:
    cuts = model._continuous_feature_quantile_cuts
    assert cuts is not None
    return [list(feature_cuts) for feature_cuts in cuts]


def test_greedy_borders_keep_every_distinct_value_of_a_skewed_feature(monkeypatch) -> None:
    x, y = _zipf_fixture()
    distinct = np.unique(x[:, 0])
    assert len(distinct) < 255, "fixture must fit the default bin budget"

    monkeypatch.delenv(EQUAL_FREQUENCY_ENV, raising=False)
    greedy = GBMRegressor(n_estimators=2).fit(x, y)
    greedy_cuts = _cuts(greedy)[0]
    assert len(greedy_cuts) == len(distinct) - 1
    expected = [float(np.float32((a + b) / 2.0)) for a, b in zip(distinct, distinct[1:])]
    assert greedy_cuts == pytest.approx(expected, rel=0, abs=0)

    monkeypatch.setenv(EQUAL_FREQUENCY_ENV, "1")
    legacy = GBMRegressor(n_estimators=2).fit(x, y)
    assert len(_cuts(legacy)[0]) < len(greedy_cuts) // 2


def test_python_mirror_matches_native_greedy_cuts(monkeypatch) -> None:
    monkeypatch.delenv(EQUAL_FREQUENCY_ENV, raising=False)
    rng = np.random.default_rng(11)
    n = 5_000
    columns = [
        rng.zipf(1.4, n).clip(1, 5_000).astype(np.float32),  # > 255 distinct
        np.where(rng.random(n) < 0.6, 0.0, rng.lognormal(size=n)).astype(np.float32),
        rng.normal(size=n).astype(np.float32),
    ]
    x = np.column_stack(columns)
    y = rng.normal(size=n).astype(np.float32)
    native = _cuts(GBMRegressor(n_estimators=1).fit(x, y))
    for feature_index, column in enumerate(columns):
        mirror = GBMRegressor._single_feature_quantile_cuts_from_sorted_values(
            sorted(float(value) for value in column), 255
        )
        assert mirror == native[feature_index], f"feature {feature_index}"
        assert len(mirror) <= 254


def test_unseen_values_fall_into_the_nearer_bin(monkeypatch) -> None:
    monkeypatch.delenv(EQUAL_FREQUENCY_ENV, raising=False)
    x = np.repeat(np.array([[0.5], [10.5]], dtype=np.float32), 50, axis=0)
    y = np.where(x[:, 0] > 5.0, 1.0, -1.0).astype(np.float32)
    model = GBMRegressor(
        n_estimators=20, training_policy="manual", min_data_in_leaf=1
    ).fit(x, y)
    assert _cuts(model) == [[5.5]]
    near_low, near_high = model.predict(np.array([[4.5], [6.5]], dtype=np.float32))
    assert near_low < 0.0 < near_high


def test_models_fitted_with_equal_frequency_cuts_predict_unchanged(monkeypatch) -> None:
    x, y = _zipf_fixture(seed=3)
    monkeypatch.setenv(EQUAL_FREQUENCY_ENV, "1")
    legacy = GBMRegressor(n_estimators=10).fit(x, y)
    expected = legacy.predict(x)
    payload = pickle.dumps(legacy)

    monkeypatch.delenv(EQUAL_FREQUENCY_ENV)
    restored = pickle.loads(payload)
    assert _cuts(restored) == _cuts(legacy)
    np.testing.assert_array_equal(restored.predict(x), expected)
